import jax
import jax.numpy as np
import optax
import argparse
import pickle
import os
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
                new_state, (logits, saccade, value, surprise) = agent_step(params, state, patches, pos_t, tasks, hp)
                return (new_state, pos_t, None), (logits, saccade, pos_t, value, surprise)
            
            final_carry, (logits_seq, saccades_seq, pos_seq, values_seq, surprise_seq) = jax.lax.scan(passive_body, carry_init, scanpaths_T)
            
            # Reshape (T, B, ...) -> (B, T, ...)
            logits_seq = np.transpose(logits_seq, (1, 0, 2))
            saccades_seq = np.transpose(saccades_seq, (1, 0, 2))
            pos_seq = np.transpose(pos_seq, (1, 0, 2))
            values_seq = np.transpose(values_seq, (1, 0)) # (B, T)
            
            # surprise_seq: (T, B, K, N, H) -> (B, T, K, N, H) -> (B, T*K, N, H)
            surprise_seq = np.transpose(surprise_seq, (1, 0, 2, 3, 4))
            B, T, K, N, H = surprise_seq.shape
            surprise_seq = surprise_seq.reshape(B, T*K, N, H)
            
            return logits_seq, saccades_seq, pos_seq, None, values_seq, surprise_seq
            
        else: # ACTIVE
            # Input is dummy range
            xs = np.arange(N_STEPS)
            
            def active_body(carry, _):
                state, pos, k = carry
                
                patches = extract_patches(images, pos, hp.patch_size)
                new_state, (logits, saccade_delta, value, surprise) = agent_step(params, state, patches, pos, tasks, hp)
                
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
                
                return (new_state, new_pos, k), (logits, saccade_delta, new_pos, log_prob, value, surprise)
            
            final_carry, (logits_seq, saccades_seq, pos_seq, log_probs_seq, values_seq, surprise_seq) = jax.lax.scan(active_body, carry_init, xs)
            
            # Reshape from (T, B, ...) to (B, T, ...)
            logits_seq = np.transpose(logits_seq, (1, 0, 2))
            saccades_seq = np.transpose(saccades_seq, (1, 0, 2))
            pos_seq = np.transpose(pos_seq, (1, 0, 2))
            log_probs_seq = np.transpose(log_probs_seq, (1, 0))
            values_seq = np.transpose(values_seq, (1, 0))
            
            # surprise_seq: (T, B, K, N, H) -> (B, T, K, N, H) -> (B, T*K, N, H)
            surprise_seq = np.transpose(surprise_seq, (1, 0, 2, 3, 4))
            B, T, K, N, H = surprise_seq.shape
            surprise_seq = surprise_seq.reshape(B, T*K, N, H)
            
            return logits_seq, saccades_seq, pos_seq, log_probs_seq, values_seq, surprise_seq
            
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

def calculate_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Calculate GAE advantages.
    rewards: (B, T)
    values: (B, T)
    Returns: advantages, targets (B, T)
    """
    # Calculate targets/adv backwards
    # next_value is 0 for the last step
    
    # We scan backwards over time.
    # Inputs to scan need to be (T, B)
    rewards_T = rewards.T
    values_T = values.T
    
    # Append value=0 for T+1
    next_values_T = np.concatenate([values_T[1:], np.zeros((1, values.shape[0]))], axis=0)
    
    deltas = rewards_T + gamma * next_values_T - values_T
    
    def scan_body(next_adv, delta):
        adv = delta + gamma * lam * next_adv
        return adv, adv
        
    _, advantages_T = jax.lax.scan(scan_body, np.zeros_like(rewards_T[0]), deltas, reverse=True)
    
    targets_T = advantages_T + values_T
    
    return advantages_T.T, targets_T.T

def make_loss_fn(rollout, n_classes, hp, aux_weight=1.0):
    
    def loss_fn(params, state, images, tasks, labels, mode, scanpaths=None, key=None, masks=None):
        logits_seq, saccades_seq, pos_seq, log_probs_seq, values_seq, surprise_seq = rollout(
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
        value_loss = 0.0
        coverage_mean = 0.0
        
        if mode == 'passive':
            # Supervised Saccade Training
            B, T, _ = pos_seq.shape
            pos_flat = pos_seq.reshape(B*T, 2)
            masks_rep = np.repeat(masks, T, axis=0)
            
            target_deltas_flat = get_oracle_saccade(pos_flat, masks_rep, None, hp.patch_size, images.shape)
            target_deltas = target_deltas_flat.reshape(B, T, 2)
            
            saccade_loss = np.mean((saccades_seq - target_deltas)**2)
            
            # Critic Training in Passive Mode?
            # We don't have rewards in passive mode really, but we can train Value to predict 0 or dummy?
            # Or just ignore Value head in passive.
            # Let's ignore Value head in passive to avoid destabilizing it with garbage.
            
            total_loss = class_loss + 1.0 * saccade_loss
            
            return total_loss, (class_loss, policy_loss, saccade_loss, acc, coverage_mean, value_loss)
            
        else: # ACTIVE
            cls_reward = (preds == labels).astype(np.float32)
            
            B, T, _ = pos_seq.shape
            masks_rep = np.repeat(masks, T, axis=0)
            pos_flat = pos_seq.reshape(B*T, 2)
            mask_patches = extract_patches(masks_rep, pos_flat, hp.patch_size)
            cov_per_step = np.mean(mask_patches, axis=(1, 2))
            cov_seq = cov_per_step.reshape(B, T)
            cov_reward = np.mean(cov_seq, axis=1)
            coverage_mean = np.mean(cov_reward)
            
            # Total Reward per step?
            # Our rewards are sparse/delayed mostly.
            # Reward structure: 
            # Step t: 5.0 * cov_reward[t]
            # Final Step: cls_reward
            
            rewards = 5.0 * cov_seq
            # Add cls_reward to last step
            rewards = rewards.at[:, -1].add(cls_reward)
            
            # GAE
            # We need to stop_gradient on values for GAE computation?
            # Yes, targets should be fixed.
            advantages, targets = calculate_gae(rewards, jax.lax.stop_gradient(values_seq))
            
            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Actor Loss (Policy Gradient)
            # log_probs: (B, T)
            # advantages: (B, T)
            # For simple A2C: -log_prob * adv
            policy_loss = -np.mean(log_probs_seq * advantages)
            
            # Critic Loss (Value)
            value_loss = np.mean((values_seq - targets)**2)
            
            # Auxiliary Supervised Loss
            target_deltas_flat = get_oracle_saccade(pos_flat, masks_rep, None, hp.patch_size, images.shape)
            target_deltas = target_deltas_flat.reshape(B, T, 2)
            aux_saccade_loss = np.mean((saccades_seq - target_deltas)**2)
            
            total_loss = class_loss + 0.1 * policy_loss + 0.5 * value_loss + aux_weight * aux_saccade_loss
            saccade_loss = aux_saccade_loss
            
        return total_loss, (class_loss, policy_loss, saccade_loss, acc, coverage_mean, value_loss)

    return loss_fn

def train_visual_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_token", type=int, default=5)
    parser.add_argument("--n_regions", type=int, default=38) # Updated default for R-Hemisphere
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--aux_weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_steps", type=int, default=15000)
    parser.add_argument("--switch_step", type=int, default=5000)
    args = parser.parse_args()
    
    # Config
    BATCH_SIZE = args.batch_size
    N_STEPS = 20
    N_TRAIN_STEPS = args.train_steps
    SWITCH_STEP = args.switch_step
    LR = args.lr
    
    # Load Connectome
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(base_dir, 'weights.txt')
    lengths_path = os.path.join(base_dir, 'tract_lengths.txt')
    
    print(f"Loading connectome from {weights_path}...")
    full_weights = onp.loadtxt(weights_path)
    full_lengths = onp.loadtxt(lengths_path)
    
    # Slice Right Hemisphere (38x38)
    n_r = 38
    weights = full_weights[:n_r, :n_r]
    lengths = full_lengths[:n_r, :n_r]
    
    # Normalize Weights
    # Divide by spectral radius or max to ensure stability?
    # Simple max normalization often sufficient for linear recurrent
    weights = weights / (np.max(weights) + 1e-8)
    
    # JAX Arrays
    weights = jax.device_put(weights)
    lengths = jax.device_put(lengths)
    
    mhsa_hp = Hyperparameters(
        n_regions=n_r, 
        n_heads=8,   
        d_k=args.d_model, d_v=args.d_model, d_model=args.d_model, 
        steps_per_token=args.steps_per_token
    )
    hp = VisualSearchHyperparameters(
        mhsa=mhsa_hp,
        patch_size=32, # Increased from 16
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
    params, state_proto = init_visual_search(hp, key, connectome_weights=weights, connectome_lengths=lengths)
    
    # Make state batch compatible
    M_batch = np.repeat(state_proto.M, BATCH_SIZE, axis=0)
    history_batch = None
    if state_proto.history is not None:
        # history: (L, B, N, D) - proto has B=1
        history_batch = np.repeat(state_proto.history, BATCH_SIZE, axis=1)
    state = NetworkState(M=M_batch, history=history_batch, step=0)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LR)
    )
    opt_state = optimizer.init(params)
    
    rollout = make_rollout(hp)
    loss_fn = make_loss_fn(rollout, hp.n_classes, hp, aux_weight=args.aux_weight)
    
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
            params, opt_state, loss, (c_loss, p_loss, s_loss, acc, cov, v_loss) = train_step_passive(
                params, opt_state, curr_state, b_imgs, b_paths, b_tasks, b_lbls, b_masks
            )
            mode = "Passive"
            print_stats = f"Loss={loss:.4f} (Cls={c_loss:.4f}, Sacc={s_loss:.4f}) | Acc={acc:.4f}"
        else:
            # Active
            params, opt_state, loss, (c_loss, p_loss, s_loss, acc, cov, v_loss) = train_step_active(
                params, opt_state, curr_state, b_imgs, b_tasks, b_lbls, b_masks, k_batch
            )
            mode = "Active"
            print_stats = f"Loss={loss:.4f} (Cls={c_loss:.4f}, Pol={p_loss:.4f}, Val={v_loss:.4f}) | Acc={acc:.4f} | Cov={cov:.4f}"
            
        if i % 500 == 0:
             print(f"Step {i:04d} [{mode}]: {print_stats}")
             
    # Save Params
    print("Saving parameters...")
    with open("visual_search_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Saved to visual_search_params.pkl")

if __name__ == "__main__":
    train_visual_search()
