import jax
import jax.numpy as np
import optax
from typing import NamedTuple
from vbjax.app.visual_search.data import generate_dataset, make_scanpaths
from vbjax.app.visual_search.model import (
    init_visual_search, agent_step, 
    VisualSearchHyperparameters, VisualSearchParams, NetworkState
)
from vbjax.ct_mhsa import Hyperparameters, init_ct_mhsa
import numpy as onp

def extract_patches(images, positions, patch_size=16):
    """
    Extract patches from images at given positions.
    images: (B, H, W, C) or (B, H, W)
    positions: (B, 2) in [-1, 1] (x, y)
    Returns: (B, patch_size, patch_size, C)
    """
    is_mask = False
    if images.ndim == 3:
        # Mask case: (B, H, W) -> (B, H, W, 1)
        images = images[..., None]
        is_mask = True
        
    B, H, W, C = images.shape
    
    # Map [-1, 1] -> [0, W]
    coords = (positions + 1) / 2 * np.array([W, H])
    
    # Center coordinates
    cx = coords[:, 0]
    cy = coords[:, 1]
    
    # Top-left corner
    start_x = np.clip(cx - patch_size // 2, 0, W - patch_size).astype(int)
    start_y = np.clip(cy - patch_size // 2, 0, H - patch_size).astype(int)
    
    def get_slice(img, sx, sy):
        return jax.lax.dynamic_slice(img, (sy, sx, 0), (patch_size, patch_size, C))
    
    patches = jax.vmap(get_slice)(images, start_x, start_y)
    
    if is_mask:
        patches = patches[..., 0]
        
    return patches

def make_rollout(hp: VisualSearchHyperparameters):
    
    def rollout_fn(params: VisualSearchParams, init_state: NetworkState, images, tasks, 
                   mode='active', scanpaths=None, key=None):
        """
        Run the agent.
        mode: 'passive' (follow scanpaths) or 'active' (use policy)
        scanpaths: (B, T, 2) required if mode='passive'
        key: required if mode='active' for sampling actions (if stochastic) or exploration noise
        """
        B = images.shape[0]
        # T is scanpaths.shape[1] if passive, else fixed N_STEPS
        N_STEPS = 10 # Fixed for active
        if mode == 'passive':
            N_STEPS = scanpaths.shape[1]
        
        # Initial Position: Center (0,0) or Random?
        # Let's start at center for active. Passive uses scanpaths[:,0].
        pos_init = np.zeros((B, 2))
        if mode == 'passive':
            pos_init = scanpaths[:, 0, :]
            
        # Carry: (state, current_pos, key)
        carry_init = (init_state, pos_init, key)
        
        # -- Implementation Split --
        
        if mode == 'passive':
            # Input is scanpaths transposed: (T, B, 2)
            scanpaths_T = np.transpose(scanpaths, (1, 0, 2))
            
            def passive_body(carry, pos_t):
                state, _, _ = carry # Ignore internal pos logic
                patches = extract_patches(images, pos_t, hp.patch_size)
                new_state, (logits, saccade) = agent_step(params, state, patches, pos_t, tasks, hp)
                return (new_state, pos_t, None), (logits, saccade, pos_t)
            
            final_carry, (logits_seq, saccades_seq, pos_seq) = jax.lax.scan(passive_body, carry_init, scanpaths_T)
            
            # Reshape (T, B, ...) -> (B, T, ...)
            logits_seq = np.transpose(logits_seq, (1, 0, 2))
            saccades_seq = np.transpose(saccades_seq, (1, 0, 2))
            pos_seq = np.transpose(pos_seq, (1, 0, 2))
            
            return logits_seq, saccades_seq, pos_seq, None
            
        else: # ACTIVE
            # Input is dummy range
            xs = np.arange(N_STEPS)
            
            def active_body(carry, _):
                state, pos, k = carry
                
                patches = extract_patches(images, pos, hp.patch_size)
                new_state, (logits, saccade_delta) = agent_step(params, state, patches, pos, tasks, hp)
                
                # Action: Saccade Delta + Noise
                # saccade_delta is in [-1, 1] via tanh
                # Add exploration noise
                k, k_noise = jax.random.split(k)
                noise = jax.random.normal(k_noise, shape=pos.shape) * 0.1 # std dev 0.1
                
                # Actual move
                move = saccade_delta + noise
                new_pos = np.clip(pos + move, -1.0, 1.0)
                
                # Compute Log Prob of the action (Gaussian)
                log_prob = -0.5 * np.sum((noise / 0.1)**2, axis=-1) # Sum over x,y
                
                return (new_state, new_pos, k), (logits, saccade_delta, new_pos, log_prob)
            
            final_carry, (logits_seq, saccades_seq, pos_seq, log_probs_seq) = jax.lax.scan(active_body, carry_init, xs)
            
            # Reshape from (T, B, ...) to (B, T, ...)
            logits_seq = np.transpose(logits_seq, (1, 0, 2))
            saccades_seq = np.transpose(saccades_seq, (1, 0, 2))
            pos_seq = np.transpose(pos_seq, (1, 0, 2))
            log_probs_seq = np.transpose(log_probs_seq, (1, 0))
            
            return logits_seq, saccades_seq, pos_seq, log_probs_seq
            
    return rollout_fn

def make_loss_fn(rollout, n_classes):
    
    def loss_fn(params, state, images, tasks, labels, mode, scanpaths=None, key=None, masks=None):
        # Run rollout
        logits_seq, saccades_seq, pos_seq, log_probs_seq = rollout(
            params, state, images, tasks, mode, scanpaths, key
        )
        
        # 1. Classification Loss (Last Step)
        final_logits = logits_seq[:, -1, :]
        one_hot = jax.nn.one_hot(labels, n_classes)
        class_loss = optax.softmax_cross_entropy(logits=final_logits, labels=one_hot).mean()
        
        # Accuracy
        preds = np.argmax(final_logits, axis=-1)
        acc = np.mean(preds == labels)
        
        # 2. Policy Loss (Active Only)
        policy_loss = 0.0
        coverage_mean = 0.0
        
        if mode == 'active':
            # Reward: 1 if correct, 0 else
            cls_reward = (preds == labels).astype(np.float32) # (B,)
            
            # Coverage Reward
            B, T, _ = pos_seq.shape
            masks_rep = np.repeat(masks, T, axis=0) # (B*T, H, W)
            pos_flat = pos_seq.reshape(B*T, 2)
            
            mask_patches = extract_patches(masks_rep, pos_flat, 16) # (B*T, 16, 16)
            
            # Coverage per step: mean of mask patch
            cov_per_step = np.mean(mask_patches, axis=(1, 2)) # (B*T,)
            cov_seq = cov_per_step.reshape(B, T)
            
            # Total Coverage per episode
            cov_reward = np.mean(cov_seq, axis=1) # (B,)
            
            coverage_mean = np.mean(cov_reward)
            
            # Total Reward: Weigh coverage to be auxiliary
            total_reward = cls_reward + 5.0 * cov_reward
            
            # Baseline (Mean reward of batch)
            baseline = np.mean(total_reward)
            advantage = total_reward - baseline
            
            # REINFORCE
            traj_log_prob = np.sum(log_probs_seq, axis=1) # (B,)
            policy_loss = -np.mean(traj_log_prob * advantage)
            
        total_loss = class_loss + 0.1 * policy_loss
        
        return total_loss, (class_loss, policy_loss, acc, coverage_mean)

    return loss_fn

def train_visual_search():
    # Config
    BATCH_SIZE = 32
    N_STEPS = 10 
    N_TRAIN_STEPS = 10000 # Increased
    SWITCH_STEP = 5000 # Increased Passive Phase
    LR = 3e-4
    
    mhsa_hp = Hyperparameters(
        n_regions=8, 
        n_heads=8,   
        d_k=16, d_v=16, d_model=32, 
        steps_per_token=1
    )
    hp = VisualSearchHyperparameters(
        mhsa=mhsa_hp,
        patch_size=16,
        n_micro_steps=3,
        n_tasks=2,
        n_classes=3,
        retina_channels=(16, 32) 
    )
    
    # Data
    print("Generating Data (with Masks)...")
    images_np, tasks_np, labels_np, masks_np = generate_dataset(n_samples=1000) 
    scanpaths_np = make_scanpaths(n_samples=1000, n_steps=N_STEPS)
    
    images = jax.device_put(images_np) / 255.0
    tasks = jax.device_put(tasks_np.flatten())
    labels = jax.device_put(labels_np)
    scanpaths = jax.device_put(scanpaths_np)
    masks = jax.device_put(masks_np)
    
    # Train/Test
    n_train = int(0.8 * len(images))
    train_imgs, test_imgs = images[:n_train], images[n_train:]
    train_paths, test_paths = scanpaths[:n_train], scanpaths[n_train:]
    train_tasks, test_tasks = tasks[:n_train], tasks[n_train:]
    train_lbls, test_lbls = labels[:n_train], labels[n_train:]
    train_masks, test_masks = masks[:n_train], masks[n_train:]
    
    # Init
    key = jax.random.PRNGKey(42)
    params, state_proto = init_visual_search(hp, key)
    
    # Make state batch compatible
    M_batch = np.repeat(state_proto.M, BATCH_SIZE, axis=0)
    history_batch = None
    if state_proto.history is not None:
        history_batch = np.repeat(state_proto.history, BATCH_SIZE, axis=1)
    state = NetworkState(M=M_batch, history=history_batch, step=0)
    
    optimizer = optax.adamw(LR)
    opt_state = optimizer.init(params)
    
    rollout = make_rollout(hp)
    loss_fn = make_loss_fn(rollout, hp.n_classes)
    
    # We need separate train steps for static arg 'mode'
    @jax.jit
    def train_step_passive(params, opt_state, state, imgs, paths, tasks, lbls):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, imgs, tasks, lbls, 'passive', paths, None, None
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    @jax.jit
    def train_step_active(params, opt_state, state, imgs, tasks, lbls, msks, key):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, imgs, tasks, lbls, 'active', None, key, msks
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    print(f"Starting Curriculum Training (Switch at {SWITCH_STEP})...")
    
    for i in range(N_TRAIN_STEPS):
        key, k_batch = jax.random.split(key)
        idx = onp.random.randint(0, len(train_imgs), BATCH_SIZE)
        
        b_imgs = train_imgs[idx]
        b_tasks = train_tasks[idx]
        b_lbls = train_lbls[idx]
        b_masks = train_masks[idx]
        
        curr_state = state # Reset M
        
        if i < SWITCH_STEP:
            # Passive
            b_paths = train_paths[idx]
            params, opt_state, loss, (c_loss, p_loss, acc, cov) = train_step_passive(
                params, opt_state, curr_state, b_imgs, b_paths, b_tasks, b_lbls
            )
            mode = "Passive"
        else:
            # Active
            params, opt_state, loss, (c_loss, p_loss, acc, cov) = train_step_active(
                params, opt_state, curr_state, b_imgs, b_tasks, b_lbls, b_masks, k_batch
            )
            mode = "Active"
            
        if i % 100 == 0:
             print(f"Step {i:04d} [{mode}]: Loss={loss:.4f} (Cls={c_loss:.4f}, Pol={p_loss:.4f}) | Acc={acc:.4f} | Cov={cov:.4f}")

if __name__ == "__main__":
    train_visual_search()