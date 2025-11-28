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

def get_oracle_saccade(pos, masks, tasks, patch_size, images_shape):
    """
    Calculate ideal saccade. Vmap over batch to ensure correct shapes.
    pos: (B, 2)
    masks: (B, H, W)
    """
    B, H, W = masks.shape
    
    # Grid (H, W, 2)
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    grid = np.stack([x_grid, y_grid], axis=-1).astype(np.float32)
    
    def single_oracle(p, m):
        # p: (2,)
        # m: (H, W)
        # grid: (H, W, 2)
        
        # Convert p to pixels
        # p is [-1, 1]
        coord = (p + 1) / 2 * np.array([W, H])
        
        # vec: (H, W, 2)
        vec = grid - coord
        
        # Weighted sum
        # m: (H, W) -> (H, W, 1)
        m_exp = m[..., None]
        
        w_vec = np.sum(vec * m_exp, axis=(0, 1)) # (2,)
        m_sum = np.sum(m) + 1e-6
        
        delta = w_vec / m_sum
        
        # Normalize
        return delta / np.array([W, H]) * 2
        
    return jax.vmap(single_oracle)(pos, masks)

def make_loss_fn(rollout, n_classes, hp):
    
    def loss_fn(params, state, images, tasks, labels, mode, scanpaths=None, key=None, masks=None):
        logits_seq, saccades_seq, pos_seq, log_probs_seq = rollout(
            params, state, images, tasks, mode, scanpaths, key
        )
        
        # 1. Classification
        final_logits = logits_seq[:, -1, :]
        one_hot = jax.nn.one_hot(labels, n_classes)
        class_loss = optax.softmax_cross_entropy(logits=final_logits, labels=one_hot).mean()
        preds = np.argmax(final_logits, axis=-1)
        acc = np.mean(preds == labels)
        
        policy_loss = 0.0
        saccade_loss = 0.0
        entropy_loss = 0.0
        coverage_mean = 0.0
        
        if mode == 'passive':
            # Supervised Saccade Training (Imitation Learning)
            # We want the network to predict the vector towards objects at each step.
            # pos_seq: (B, T, 2) - These are the FORCED positions.
            # saccades_seq: (B, T, 2) - The network's PREDICTED output at that forced position.
            
            # Calculate Oracle Target for each step
            B, T, _ = pos_seq.shape
            
            # Flatten for batch processing
            pos_flat = pos_seq.reshape(B*T, 2)
            masks_rep = np.repeat(masks, T, axis=0) # (B*T, H, W)
            # tasks_rep = np.repeat(tasks, T, axis=0)
            
            target_deltas_flat = get_oracle_saccade(pos_flat, masks_rep, None, hp.patch_size, images.shape)
            target_deltas = target_deltas_flat.reshape(B, T, 2)
            
            # MSE Loss
            saccade_loss = np.mean((saccades_seq - target_deltas)**2)
            
            # Total Passive Loss
            total_loss = class_loss + 1.0 * saccade_loss
            
        else: # active
            # ... (Existing Active Logic) ...
            cls_reward = (preds == labels).astype(np.float32)
            
            B, T, _ = pos_seq.shape
            masks_rep = np.repeat(masks, T, axis=0)
            pos_flat = pos_seq.reshape(B*T, 2)
            mask_patches = extract_patches(masks_rep, pos_flat, hp.patch_size)
            cov_per_step = np.mean(mask_patches, axis=(1, 2))
            cov_seq = cov_per_step.reshape(B, T)
            cov_reward = np.mean(cov_seq, axis=1)
            coverage_mean = np.mean(cov_reward)
            
            total_reward = cls_reward + 5.0 * cov_reward
            baseline = np.mean(total_reward)
            advantage = total_reward - baseline
            
            traj_log_prob = np.sum(log_probs_seq, axis=1)
            policy_loss = -np.mean(traj_log_prob * advantage)
            
            # Entropy Regularization
            # We don't have the full distribution, just the sample log_prob.
            # For Gaussian policy with fixed std (0.1), entropy is constant constant + log(std).
            # BUT, if we learned sigma, we'd maximize it.
            # Since sigma is fixed/noise is added externally, "entropy" here effectively means
            # maximizing the spread of actions if we were outputting logits.
            # With deterministic Tanh output + Noise, the "Policy" is N(NetworkOut, FixedSigma).
            # The entropy of this fixed-sigma Gaussian is constant.
            
            # HOWEVER, usually in continuous control, we want to penalize "certainty" or saturate tanh.
            # Or, we just rely on the noise.
            # If we want "Entropy", we usually mean "Exploration Bonus".
            # Since we inject fixed noise, we are enforcing exploration.
            # Maybe we don't need explicit entropy term if sigma is fixed?
            # Correct. Fixed sigma = fixed entropy.
            
            # Let's stick to just the Coverage Reward for now, as "Entropy" is implicitly fixed by the noise injection.
            # Unless we want to penalize the magnitude of the mean (regularization)? No.
            
            total_loss = class_loss + 0.1 * policy_loss
            
        return total_loss, (class_loss, policy_loss, saccade_loss, acc, coverage_mean)

    return loss_fn

def train_visual_search():
    # Config
    BATCH_SIZE = 32
    N_STEPS = 20
    N_TRAIN_STEPS = 20000 # Increased
    SWITCH_STEP = 5000 # Increased Passive Phase
    LR = 1e-4
    
    mhsa_hp = Hyperparameters(
        n_regions=8, 
        n_heads=8,   
        d_k=32, d_v=32, d_model=32, 
        steps_per_token=1
    )
    hp = VisualSearchHyperparameters(
        mhsa=mhsa_hp,
        patch_size=32, # Increased from 16
        n_micro_steps=5, # Increased from 3
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
    loss_fn = make_loss_fn(rollout, hp.n_classes, hp)
    
    # We need separate train steps for static arg 'mode'
    @jax.jit
    def train_step_passive(params, opt_state, state, imgs, paths, tasks, lbls, msks):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, imgs, tasks, lbls, 'passive', paths, None, msks
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
            params, opt_state, loss, (c_loss, p_loss, s_loss, acc, cov) = train_step_passive(
                params, opt_state, curr_state, b_imgs, b_paths, b_tasks, b_lbls, b_masks
            )
            mode = "Passive"
            print_stats = f"Loss={loss:.4f} (Cls={c_loss:.4f}, Sacc={s_loss:.4f}) | Acc={acc:.4f}"
        else:
            # Active
            params, opt_state, loss, (c_loss, p_loss, s_loss, acc, cov) = train_step_active(
                params, opt_state, curr_state, b_imgs, b_tasks, b_lbls, b_masks, k_batch
            )
            mode = "Active"
            print_stats = f"Loss={loss:.4f} (Cls={c_loss:.4f}, Pol={p_loss:.4f}) | Acc={acc:.4f} | Cov={cov:.4f}"
            
        if i % 100 == 0:
             print(f"Step {i:04d} [{mode}]: {print_stats}")

if __name__ == "__main__":
    train_visual_search()
