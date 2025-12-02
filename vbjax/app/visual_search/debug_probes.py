import jax
import jax.numpy as np
import optax
import numpy as onp
import os
import pickle
from vbjax.app.visual_search.model import init_visual_search, VisualSearchHyperparameters, agent_step, NetworkState
from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.visual_search.data import generate_dataset, make_scanpaths
from vbjax.app.visual_search.train import make_rollout, make_loss_fn

def setup_debug_environment(batch_size=4):
    # Load Connectome
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(base_dir, 'weights.txt')
    lengths_path = os.path.join(base_dir, 'tract_lengths.txt')
    
    full_weights = onp.loadtxt(weights_path)
    full_lengths = onp.loadtxt(lengths_path)
    n_r = 38
    weights = full_weights[:n_r, :n_r] / (np.max(full_weights[:n_r, :n_r]) + 1e-8)
    lengths = full_lengths[:n_r, :n_r]
    
    weights = jax.device_put(weights)
    lengths = jax.device_put(lengths)
    
    mhsa_hp = Hyperparameters(n_regions=n_r, n_heads=8, d_k=32, d_v=32, d_model=32, steps_per_token=5)
    hp = VisualSearchHyperparameters(mhsa=mhsa_hp, patch_size=32, n_tasks=2, n_classes=3, retina_channels=(16, 32))
    
    # Init State
    key = jax.random.PRNGKey(42)
    params, state_proto = init_visual_search(hp, key, connectome_weights=weights, connectome_lengths=lengths)
    
    M_batch = np.repeat(state_proto.M, batch_size, axis=0)
    history_batch = None
    if state_proto.history is not None:
        history_batch = np.repeat(state_proto.history, batch_size, axis=1)
    state = NetworkState(M=M_batch, history=history_batch, step=0)
    
    return params, state, hp, weights, lengths

def probe_1_overfitting():
    print("\n=== Probe 1: One-Step Overfitting (Single Batch) ===")
    BATCH_SIZE = 4
    N_STEPS = 100
    LR = 1e-3 # High LR for fast overfitting
    
    params, state, hp, _, _ = setup_debug_environment(BATCH_SIZE)
    
    # Generate fixed batch
    images_np, tasks_np, labels_np, masks_np = generate_dataset(n_samples=BATCH_SIZE)
    scanpaths_np = make_scanpaths(n_samples=BATCH_SIZE, n_steps=30)
    
    images = jax.device_put(images_np) / 255.0
    tasks = jax.device_put(tasks_np.flatten())
    labels = jax.device_put(labels_np)
    scanpaths = jax.device_put(scanpaths_np)
    masks = jax.device_put(masks_np)
    
    # Rollout & Loss
    rollout = make_rollout(hp, n_steps=30)
    loss_fn_base = make_loss_fn(rollout, hp.n_classes, hp, cls_mask_steps=0)
    
    def loss_fn(params, state, images, tasks, labels, scanpaths):
        # Reduced args for debug
        return loss_fn_base(params, state, images, tasks, labels, 'passive', scanpaths, None, masks, 1.0)

    optimizer = optax.adamw(LR)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, images, tasks, labels, scanpaths)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux
        
    print(f"Training on single batch of {BATCH_SIZE} for {N_STEPS} steps...")
    for i in range(N_STEPS):
        params, opt_state, loss, aux = train_step(params, opt_state, state)
        acc = aux[3]
        if i % 10 == 0:
            print(f"Step {i}: Loss={loss:.4f}, Acc={acc:.2%}")
            
    print(f"Final Accuracy: {acc:.2%}")
    if acc > 0.99:
        print("PASS: Model can memorize a single batch.")
    else:
        print("FAIL: Model cannot memorize a single batch. Architecture or Code is broken.")

def probe_2_input_sensitivity():
    print("\n=== Probe 2: Input Sensitivity (Gradient w.r.t Image) ===")
    BATCH_SIZE = 1
    params, state, hp, _, _ = setup_debug_environment(BATCH_SIZE)
    
    # Data: Random Noise
    key = jax.random.PRNGKey(123)
    images = jax.random.uniform(key, (BATCH_SIZE, 128, 128, 3))
    
    # Generate dummy tasks/paths
    tasks = jax.numpy.zeros((BATCH_SIZE,), dtype=int)
    # Force a scanpath that is definitely inside the image
    scanpaths = jax.numpy.zeros((BATCH_SIZE, 10, 2)) # Center (0,0)
    
    # 2a. Check Retina Gradient
    print("--- 2a. Checking Retina Gradient ---")
    from vbjax.app.visual_search.train import extract_patches
    from vbjax.app.visual_search.model import retina_forward
    
    def retina_loss(images_in):
        pos_0 = scanpaths[:, 0, :]
        patches = extract_patches(images_in, pos_0, hp.patch_size)
        vis_feat = retina_forward(params, patches)
        return np.sum(vis_feat)
        
    grad_retina = jax.grad(retina_loss)(images)
    print(f"Retina Grad Norm: {np.linalg.norm(grad_retina):.6f}")

    # 2b. Full Rollout Gradient
    print("--- 2b. Checking Full Rollout Gradient ---")
    rollout = make_rollout(hp, n_steps=10)
    
    def forward_logit_sum(images_in):
        # Run rollout
        logits_seq, _, _, _, _, _, _ = rollout(
            params, state, images_in, tasks, 'passive', scanpaths, None
        )
        # Sum of all logits
        return np.sum(logits_seq)
        
    # Compute Gradient w.r.t Images
    grad_fn = jax.grad(forward_logit_sum)
    grads = grad_fn(images)
    
    grad_norm = np.linalg.norm(grads)
    
    print(f"Full Grad Norm: {grad_norm:.6f}")
    
    if grad_norm > 1e-6:
        print("PASS: Non-zero gradient w.r.t input.")
    else:
        print("FAIL: Zero gradient w.r.t input.")

def probe_3_trace_signal():
    print("\n=== Probe 3: Signal Trace (Forward Pass) ===")
    BATCH_SIZE = 1
    params, state, hp, _, _ = setup_debug_environment(BATCH_SIZE)
    
    images_np, tasks_np, labels_np, masks_np = generate_dataset(n_samples=BATCH_SIZE)
    scanpaths_np = make_scanpaths(n_samples=BATCH_SIZE, n_steps=1)
    
    images = jax.device_put(images_np) / 255.0
    pos = jax.device_put(scanpaths_np[:, 0, :])
    task_idx = jax.device_put(tasks_np.flatten())
    
    print(f"Images Stats: Min={np.min(images):.4f}, Max={np.max(images):.4f}, Mean={np.mean(images):.4f}, Norm={np.linalg.norm(images):.4f}")
    
    # 1. Extract Patches
    from vbjax.app.visual_search.train import extract_patches
    
    # Debug Coords
    B, H, W, C = images.shape
    coords = (pos + 1) / 2 * np.array([W, H])
    print(f"Pos Raw: {pos[0]}")
    print(f"Coords: {coords[0]}")
    cx, cy = coords[0]
    sx = int(cx - hp.patch_size // 2)
    sy = int(cy - hp.patch_size // 2)
    print(f"Slice Start: x={sx}, y={sy}")
    
    patches = extract_patches(images, pos, hp.patch_size)
    print(f"Patches Norm: {np.linalg.norm(patches):.4f}")
    print(f"Patches Min: {np.min(patches):.4f}, Max: {np.max(patches):.4f}")
    
    if np.linalg.norm(patches) == 0:
        print("DEBUG: Dumping Image stats at slice location")
        # Extract manually using numpy to verify
        img_np = np.array(images[0])
        patch_np = img_np[sy:sy+hp.patch_size, sx:sx+hp.patch_size, :]
        print(f"Manual Patch Norm: {onp.linalg.norm(patch_np):.4f}")
        print(f"Image Center Pixel: {img_np[int(cy), int(cx)]}")

    
    # 2. Retina Forward
    from vbjax.app.visual_search.model import retina_forward
    vis_feat = retina_forward(params, patches)
    print(f"VisFeat Norm: {np.linalg.norm(vis_feat):.4f}")
    
    # 3. Embeddings
    pos_feat = pos @ params.pos_embed_w + params.pos_embed_b
    task_feat = params.task_embed[task_idx]
    print(f"PosFeat Norm: {np.linalg.norm(pos_feat):.4f}")
    print(f"TaskFeat Norm: {np.linalg.norm(task_feat):.4f}")
    
    # 4. Agent Step (Manual Unroll)
    from vbjax.app.visual_search.model import IDX_R_V1, IDX_R_PFC
    
    # Reconstruct agent_step logic
    input_feat = vis_feat + pos_feat # Original logic
    # input_feat = vis_feat + pos_feat + task_feat # If using continuous task
    
    core_input = np.zeros((BATCH_SIZE, hp.mhsa.n_regions, hp.mhsa.d_model))
    core_input = core_input.at[:, IDX_R_V1, :].set(input_feat)
    core_input = core_input.at[:, IDX_R_PFC, :].set(task_feat)
    
    print(f"CoreInput Norm: {np.linalg.norm(core_input):.4f}")
    print(f"CoreInput V1 Norm: {np.linalg.norm(core_input[:, IDX_R_V1, :]):.4f}")
    
    # 5. MHSA Step
    from vbjax.ct_mhsa import mhsa_step
    new_state, y, surprise = mhsa_step(params.core, state, core_input, hp.mhsa)
    
    print(f"MHSA Output y Norm: {np.linalg.norm(y):.4f}")
    print(f"y at V1 Norm: {np.linalg.norm(y[:, IDX_R_V1, :]):.4f}")
    print(f"y at PFC Norm: {np.linalg.norm(y[:, IDX_R_PFC, :]):.4f}")
    
    # 6. Readout
    from vbjax.app.visual_search.model import IDX_R_PFC
    pfc_activity = y[:, IDX_R_PFC, :]
    logits = pfc_activity @ params.head_answer_w + params.head_answer_b
    print(f"Logits Norm: {np.linalg.norm(logits):.4f}")

def probe_4_agent_grad():
    print("\n=== Probe 4: Agent Step Gradient Isolation ===")
    BATCH_SIZE = 1
    params, state, hp, _, _ = setup_debug_environment(BATCH_SIZE)
    
    key = jax.random.PRNGKey(123)
    patch = jax.random.uniform(key, (BATCH_SIZE, hp.patch_size, hp.patch_size, 3))
    pos = jax.numpy.zeros((BATCH_SIZE, 2))
    task_idx = jax.numpy.zeros((BATCH_SIZE,), dtype=int)
    step_idx = 0
    
    # Random projection for loss
    key, k_p = jax.random.split(key)
    proj = jax.random.normal(k_p, (hp.n_classes,))

    def agent_loss(p_in):
        # Use fixed agent_step
        new_state, (logits, saccade, value, surprise, priority) = agent_step(
            params, state, p_in, pos, task_idx, step_idx, hp
        )
        return np.sum(logits * proj)
        
    grad_fn = jax.grad(agent_loss)
    grad = grad_fn(patch)
    
    print(f"Agent Step Grad Norm: {np.linalg.norm(grad):.6f}")

def probe_5_mhsa_grad():
    print("\n=== Probe 5: MHSA Step Gradient Isolation ===")
    BATCH_SIZE = 1
    params, state, hp, _, _ = setup_debug_environment(BATCH_SIZE)
    
    # Create random input
    key = jax.random.PRNGKey(42)
    core_input = jax.random.normal(key, (BATCH_SIZE, hp.mhsa.n_regions, hp.mhsa.d_model))
    
    # Random projection
    key, k_p = jax.random.split(key)
    proj = jax.random.normal(k_p, (BATCH_SIZE, hp.mhsa.n_regions, hp.mhsa.d_model))
    
    def mhsa_loss(x_in):
        from vbjax.ct_mhsa import mhsa_step
        new_state, y, surprise = mhsa_step(params.core, state, x_in, hp.mhsa)
        return np.sum(y * proj)
        
    grad_fn = jax.grad(mhsa_loss)
    grad = grad_fn(core_input)
    
    print(f"MHSA Grad Norm: {np.linalg.norm(grad):.6f}")

if __name__ == "__main__":
    probe_1_overfitting()
    probe_2_input_sensitivity()
    probe_3_trace_signal()
    probe_4_agent_grad()
    probe_5_mhsa_grad()
