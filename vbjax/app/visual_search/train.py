import jax
import jax.numpy as np
import optax
import argparse
import pickle
import os
from typing import NamedTuple
from vbjax.app.visual_search.data import generate_dataset, make_scanpaths
from vbjax.app.visual_search.model import (
    init_visual_search, agent_step, agent_step_eval, 
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

def get_target_coords(masks):
    """
    Compute center of mass of masks to get target coordinates.
    masks: (B, H, W)
    Returns: (B, 2) in [-1, 1]
    """
    B, H, W = masks.shape
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Weighted sum
    # masks sum might be 0 if no target (shouldn't happen in this dataset)
    m_sum = np.sum(masks, axis=(1, 2)) + 1e-6
    
    cy = np.sum(masks * y_grid, axis=(1, 2)) / m_sum
    cx = np.sum(masks * x_grid, axis=(1, 2)) / m_sum
    
    # Normalize to [-1, 1]
    # 0 -> -1, W -> 1
    cx_norm = (cx / W) * 2 - 1
    cy_norm = (cy / H) * 2 - 1
    
    return np.stack([cx_norm, cy_norm], axis=-1)

def make_rollout(hp: VisualSearchHyperparameters, n_steps=10, deterministic=False):
    
    def rollout_fn(params: VisualSearchParams, init_state: NetworkState, images, tasks, 
                   mode='active', scanpaths=None, key=None):
        """
        Run the agent.
        """
        B = images.shape[0]
        N_STEPS = n_steps # Use passed n_steps
        if mode == 'passive':
            N_STEPS = scanpaths.shape[1]
        
        # Initial Position
        pos_init = np.zeros((B, 2))
        if mode == 'passive':
            pos_init = scanpaths[:, 0, :]
            
        # Carry: (state, current_pos, key)
        carry_init = (init_state, pos_init, key)
        
        # -- Implementation Split --
        
        if mode == 'passive':
            # Input is scanpaths transposed: (T, B, 2)
            scanpaths_T = np.transpose(scanpaths, (1, 0, 2))
            steps = np.arange(N_STEPS)
            
            def passive_body(carry, inputs):
                pos_t, step_t = inputs
                state, _, _ = carry # Ignore internal pos logic
                patches = extract_patches(images, pos_t, hp.patch_size)
                new_state, (logits, saccade, value, surprise, priority) = agent_step(params, state, patches, pos_t, tasks, step_t, hp)
                return (new_state, pos_t, None), (logits, saccade, pos_t, value, surprise, priority)
            
            final_carry, (logits_seq, saccades_seq, pos_seq, values_seq, surprise_seq, priority_seq) = jax.lax.scan(passive_body, carry_init, (scanpaths_T, steps))
            
            # Reshape (T, B, ...) -> (B, T, ...)
            logits_seq = np.transpose(logits_seq, (1, 0, 2))
            saccades_seq = np.transpose(saccades_seq, (1, 0, 2))
            pos_seq = np.transpose(pos_seq, (1, 0, 2))
            values_seq = np.transpose(values_seq, (1, 0)) # (B, T)
            priority_seq = np.transpose(priority_seq, (1, 0)) # (B, T)
            
            # surprise_seq: (T, B, K, N, H) -> (B, T, K, N, H) -> (B, T*K, N, H)
            surprise_seq = np.transpose(surprise_seq, (1, 0, 2, 3, 4))
            B, T, K, N, H = surprise_seq.shape
            surprise_seq = surprise_seq.reshape(B, T*K, N, H)
            
            return logits_seq, saccades_seq, pos_seq, None, values_seq, surprise_seq, priority_seq
            
        else: # ACTIVE
            # Input is dummy range
            xs = np.arange(N_STEPS)
            
            def active_body(carry, step_t):
                state, pos, k = carry
                
                patches = extract_patches(images, pos, hp.patch_size)
                
                if deterministic:
                    new_state, (logits, saccade_delta, value, surprise, priority) = agent_step_eval(params, state, patches, pos, tasks, step_t, hp)
                    move = saccade_delta # No noise
                    log_prob = np.zeros(B) # No log_prob for deterministic policy
                else:
                    new_state, (logits, saccade_delta, value, surprise, priority) = agent_step(params, state, patches, pos, tasks, step_t, hp)
                    # Action: Saccade Delta + Noise
                    # saccade_delta is in [-1, 1] via tanh
                    # Add exploration noise
                    k, k_noise = jax.random.split(k)
                    noise = jax.random.normal(k_noise, shape=pos.shape) * 0.1 # std dev 0.1
                    
                    # Actual move
                    move = saccade_delta + noise
                    
                    # Compute Log Prob of the action (Gaussian)
                    log_prob = -0.5 * np.sum((noise / 0.1)**2, axis=-1) # Sum over x,y
                
                new_pos = np.clip(pos + move, -1.0, 1.0)
                
                return (new_state, new_pos, k), (logits, saccade_delta, new_pos, log_prob, value, surprise, priority)
            
            final_carry, (logits_seq, saccades_seq, pos_seq, log_probs_seq, values_seq, surprise_seq, priority_seq) = jax.lax.scan(active_body, carry_init, xs)
            
            # Reshape from (T, B, ...) to (B, T, ...)
            logits_seq = np.transpose(logits_seq, (1, 0, 2))
            saccades_seq = np.transpose(saccades_seq, (1, 0, 2))
            pos_seq = np.transpose(pos_seq, (1, 0, 2))
            log_probs_seq = np.transpose(log_probs_seq, (1, 0))
            values_seq = np.transpose(values_seq, (1, 0))
            priority_seq = np.transpose(priority_seq, (1, 0))
            
            # surprise_seq: (T, B, K, N, H) -> (B, T, K, N, H) -> (B, T*K, N, H)
            surprise_seq = np.transpose(surprise_seq, (1, 0, 2, 3, 4))
            B, T, K, N, H = surprise_seq.shape
            surprise_seq = surprise_seq.reshape(B, T*K, N, H)
            
            return logits_seq, saccades_seq, pos_seq, log_probs_seq, values_seq, surprise_seq, priority_seq
            
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
        # p: (2,) - Current eye position in [-1, 1]
        # m: (H, W) - Binary mask of objects
        
        # Convert p to pixels
        # p is [-1, 1] -> [0, W]
        coord = (p + 1) / 2 * np.array([W, H]) # (2,)
        
        # Grid: (H, W, 2)
        # We need distances from coord to every pixel
        diff = grid - coord # (H, W, 2)
        dist_sq = np.sum(diff**2, axis=-1) # (H, W)
        
        # Mask out invalid pixels (where m == 0)
        # Add a large value to dist_sq where m is 0
        # m is usually 0.0 or 1.0
        valid_dist = dist_sq + (1.0 - m) * 1e9
        
        # Find argmin (nearest active pixel)
        # We flatten to find index
        flat_dist = valid_dist.ravel()
        min_idx = np.argmin(flat_dist)
        
        # Get vector to that pixel
        # We can re-use diff
        flat_diff = diff.reshape(-1, 2)
        target_vec = flat_diff[min_idx] # (2,)
        
        # Normalize to [-1, 1] coordinate space delta
        # Pixel delta -> Normalized delta
        # 2 * delta / Size
        norm_delta = target_vec / np.array([W, H]) * 2
        
        # If mask is empty (all 0), m_sum is 0.
        # Check if mask has any targets
        has_target = np.max(m) > 0.5
        
        # If no target, return 0,0
        return jax.lax.select(has_target, norm_delta, np.zeros(2))
        
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

def make_loss_fn(rollout, n_classes, hp, term_reward=10.0, shape_reward=5.0, cls_mask_steps=0):
    
    def loss_fn(params, state, images, tasks, labels, mode, scanpaths=None, key=None, masks=None, aux_weight=1.0):
        logits_seq, saccades_seq, pos_seq, log_probs_seq, values_seq, surprise_seq, priority_seq = rollout(
            params, state, images, tasks, mode, scanpaths, key
        )
        
        # Get Target Coords
        target_coords = get_target_coords(masks) # (B, 2)
        
        # Calculate Distances over time
        # pos_seq: (B, T, 2)
        # target: (B, 1, 2)
        dists = np.sqrt(np.sum((pos_seq - target_coords[:, None, :])**2, axis=-1)) # (B, T)
        
        # 1. Dense Classification (PFCDL Supervision)
        # logits_seq: (B, T, n_classes)
        one_hot = jax.nn.one_hot(labels, n_classes) # (B, n_classes)
        # Broadcast labels to (B, T, n_classes)
        one_hot_seq = np.repeat(one_hot[:, None, :], logits_seq.shape[1], axis=1)
        
        # Masked Classification Loss
        # Create Mask: 0 for t < cls_mask_steps, 1 otherwise
        B, T, _ = logits_seq.shape
        time_indices = np.arange(T)
        loss_mask = (time_indices >= cls_mask_steps).astype(np.float32) # (T,)
        loss_mask = loss_mask[None, :] # (1, T)
        
        per_step_loss = optax.softmax_cross_entropy(logits=logits_seq, labels=one_hot_seq) # (B, T)
        
        # Apply mask and normalize by the effective number of steps
        masked_loss = per_step_loss * loss_mask
        dense_cls_loss = np.sum(masked_loss) / (np.sum(loss_mask) * B + 1e-8)
        
        class_loss = dense_cls_loss # Use dense loss as main class loss
        
        # Final Accuracy
        final_logits = logits_seq[:, -1, :]
        preds = np.argmax(final_logits, axis=-1)
        acc = np.mean(preds == labels)
        
        
        # 2. Priority Supervision (PCIP)
        # Target: 1.0 - dist (closer = higher priority)
        # Clip to [0, 1]
        priority_target = np.clip(1.0 - dists, 0.0, 1.0)
        priority_loss = np.mean((priority_seq - priority_target)**2)
        
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
            
            total_loss = dense_cls_loss + 1.0 * saccade_loss + 0.5 * priority_loss
            
            return total_loss, (class_loss, policy_loss, saccade_loss, acc, coverage_mean, value_loss, priority_loss)
            
        else: # ACTIVE
            cls_reward = (preds == labels).astype(np.float32)
            
            B, T, _ = pos_seq.shape
            masks_rep = np.repeat(masks, T, axis=0)
            # pos_flat = pos_seq.reshape(B*T, 2)
            # mask_patches = extract_patches(masks_rep, pos_flat, hp.patch_size)
            # cov_per_step = np.mean(mask_patches, axis=(1, 2))
            # cov_seq = cov_per_step.reshape(B, T)
            # cov_reward = np.mean(cov_seq, axis=1)
            # coverage_mean = np.mean(cov_reward)
            
            # Reward Shaping (Potential Based)
            # R_shape = dist_{t-1} - dist_t
            # dists: (B, T).
            # Assume dist_{-1} is initial dist (based on pos_init which was 0,0)
            # Let's use dists[:, 0] as baseline for step 0
            dist_diff = -1.0 * (dists[:, 1:] - dists[:, :-1]) # (B, T-1) positive if closer
            # Pad with 0 for first step or last?
            # Rewards align with actions 0..T-1.
            shaping_rewards = np.concatenate([np.zeros((B, 1)), dist_diff], axis=1)
            
            # Updated Rewards: Remove Coverage, Increase Shaping/Terminal
            # rewards = 5.0 * cov_seq + 2.0 * shaping_rewards
            rewards = shape_reward * shaping_rewards
            # Add cls_reward to last step
            rewards = rewards.at[:, -1].add(term_reward * cls_reward)
            
            # GAE
            advantages, targets = calculate_gae(rewards, jax.lax.stop_gradient(values_seq))
            
            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Actor Loss (Policy Gradient)
            policy_loss = -np.mean(log_probs_seq * advantages)
            
            # Critic Loss (Value)
            value_loss = np.mean((values_seq - targets)**2)
            
            # Auxiliary Supervised Loss
            target_deltas_flat = get_oracle_saccade(pos_seq.reshape(B*T, 2), np.repeat(masks, T, axis=0), None, hp.patch_size, images.shape)
            target_deltas = target_deltas_flat.reshape(B, T, 2)
            aux_saccade_loss = np.mean((saccades_seq - target_deltas)**2)
            
            # Entropy Regularization? (Not implemented yet, but good to have)
            
            total_loss = dense_cls_loss + 0.1 * policy_loss + 0.5 * value_loss + aux_weight * aux_saccade_loss + 0.5 * priority_loss
            saccade_loss = aux_saccade_loss
            
        return total_loss, (class_loss, policy_loss, saccade_loss, acc, coverage_mean, value_loss, priority_loss)

    return loss_fn

def train_visual_search():
    """
    Trains the Cortico-Thalamic MHSA model for the Visual Search task.

    Recommended Training Approach for Optimal Performance and Reproducibility:

    Phase 1: Saccade Policy Warm-up (Supervised Learning)
    -----------------------------------------------------
    First, train the agent's Frontal Eye Field (FEF) to accurately target objects.
    Use `vbjax/app/visual_search/train_saccade.py` with default parameters.
    This script will save a `visual_search_params.pkl` checkpoint containing the warm-started FEF weights.
    Example command:
    `python3 -m vbjax.app.visual_search.train_saccade --steps 2000 --batch_size 32 --lr 1e-3`
    This phase ensures the agent can reliably move its gaze towards salient objects before engaging in full reinforcement learning.

    Phase 2: Active Visual Search (Reinforcement Learning with Stable Saccade Supervision)
    --------------------------------------------------------------------------------------
    After warming up the FEF, switch to active training mode using this `train.py` script.
    It is crucial to load the checkpoint from Phase 1 and maintain a constant, significant
    `aux_weight` for saccade supervision throughout this phase. This prevents the FEF policy
    from degrading under the exploration pressures of reinforcement learning.

    Example command:
    `python3 -m vbjax.app.visual_search.train --train_steps 30000 --batch_size 32 --n_steps 30 \
        --switch_step 0 --checkpoint visual_search_params.pkl --aux_weight 1.0`

    Parameters:
        --batch_size (int): Number of episodes per training step.
        --steps_per_token (int): Number of micro-steps for the MHSA core per time step.
        --n_regions (int): Number of regions in the connectome (default 38 for right hemisphere).
        --d_model (int): Dimensionality of the model's internal representations.
        --aux_weight (float): Weight for the auxiliary saccade loss during active training.
                              Keep at a constant value (e.g., 1.0) for stable FEF guidance.
        --lr (float): Learning rate for the AdamW optimizer.
        --train_steps (int): Total number of training steps.
        --switch_step (int): Step at which to switch from passive to active mode. Set to 0 to start directly in active mode.
        --n_steps (int): Number of time steps (fixations) per episode.
        --terminal_reward (float): Reward granted for correct classification at the final step.
        --shaping_reward (float): Reward for approaching the target object.
        --cls_mask_steps (int): Number of initial steps to mask classification loss (e.g., to allow exploration).
        --checkpoint (str): Path to a pre-trained model checkpoint (e.g., `visual_search_params.pkl`).
                            Highly recommended to use the output of `train_saccade.py`.

    The training process saves the final model parameters to `visual_search_params.pkl`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_token", type=int, default=5)
    parser.add_argument("--n_regions", type=int, default=38) # Updated default for R-Hemisphere
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--aux_weight", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_steps", type=int, default=60000)
    parser.add_argument("--switch_step", type=int, default=15000) # Ensure it stays passive
    parser.add_argument("--n_steps", type=int, default=30) # New arg
    parser.add_argument("--terminal_reward", type=float, default=10.0)
    parser.add_argument("--shaping_reward", type=float, default=5.0)
    parser.add_argument("--cls_mask_steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    # Config
    BATCH_SIZE = args.batch_size
    N_STEPS = args.n_steps
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
    weights = weights / (np.max(weights) + 1e-8)
    
    # JAX Arrays
    weights = jax.device_put(weights)
    lengths = jax.device_put(lengths)
    
    mhsa_hp = Hyperparameters(
        n_regions=n_r, 
        n_heads=8,   
        d_k=16, d_v=16, d_model=args.d_model, 
        steps_per_token=args.steps_per_token
    )
    hp = VisualSearchHyperparameters(
        mhsa=mhsa_hp,
        patch_size=64, # Increased from 32
        n_tasks=2,
        n_classes=3,
        retina_channels=(16, 32) 
    )
    
    # Data - Removed static generation
    
    # Init
    key = jax.random.PRNGKey(42)
    params, state_proto = init_visual_search(hp, key, connectome_weights=weights, connectome_lengths=lengths)
    
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}...")
            with open(args.checkpoint, "rb") as f:
                loaded_params = pickle.load(f)
            # We assume structure matches. JAX doesn't check strict structure on replace usually, but let's be safe.
            # Ideally we traverse tree. For now, direct assignment.
            params = loaded_params
        else:
            print(f"Checkpoint {args.checkpoint} not found. Starting from scratch.")

    # Make state batch compatible
    M_batch = np.repeat(state_proto.M, BATCH_SIZE, axis=0)
    history_batch = None
    if state_proto.history is not None:
        # history: (L, B, N, D) - proto has B=1
        history_batch = np.repeat(state_proto.history, BATCH_SIZE, axis=1)
    # Lags handled in init now
    lags = state_proto.lags
    state = NetworkState(M=M_batch, history=history_batch, step=0, lags=lags)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LR)
    )
    opt_state = optimizer.init(params)
    
    rollout = make_rollout(hp, n_steps=N_STEPS) # Pass N_STEPS
    loss_fn = make_loss_fn(rollout, hp.n_classes, hp, term_reward=args.terminal_reward, shape_reward=args.shaping_reward, cls_mask_steps=args.cls_mask_steps)
    
    # We need separate train steps for static arg 'mode'
    @jax.jit
    def train_step_passive(params, opt_state, state, imgs, paths, tasks, lbls, msks, aux_w):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, imgs, tasks, lbls, 'passive', paths, None, msks, aux_w
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    @jax.jit
    def train_step_active(params, opt_state, state, imgs, tasks, lbls, msks, key, aux_w):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, imgs, tasks, lbls, 'active', None, key, msks, aux_w
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    print(f"Starting Curriculum Training (Switch at {SWITCH_STEP})...")
    
    ANNEAL_STEPS = 2000
    
    for i in range(N_TRAIN_STEPS):
        key, k_batch, k_paths = jax.random.split(key, 3)
        
        # Generate Fresh Data
        b_imgs_np, b_tasks_np, b_lbls_np, b_masks_np = generate_dataset(n_samples=BATCH_SIZE, seed=onp.random.randint(0, 1000000))
        
        b_imgs = jax.device_put(b_imgs_np) / 255.0
        b_tasks = jax.device_put(b_tasks_np.flatten())
        b_lbls = jax.device_put(b_lbls_np)
        b_masks = jax.device_put(b_masks_np)
        
        curr_state = state # Reset M
        
        if i < SWITCH_STEP:
            # Passive - Generate Random Scanpaths
            b_paths = jax.random.uniform(k_paths, (BATCH_SIZE, N_STEPS, 2), minval=-0.8, maxval=0.8)
            
            params, opt_state, loss, (c_loss, p_loss, s_loss, acc, cov, v_loss, pri_loss) = train_step_passive(
                params, opt_state, curr_state, b_imgs, b_paths, b_tasks, b_lbls, b_masks, 1.0
            )
            mode = "Passive"
            print_stats = f"Loss={loss:.4f} (Cls={c_loss:.4f}, Sacc={s_loss:.4f}, Pri={pri_loss:.4f}) | Acc={acc:.4f}"
        else: # Active
            # Maintain Aux Weight (no annealing)
            current_aux_weight = args.aux_weight
            
            params, opt_state, loss, (c_loss, p_loss, s_loss, acc, cov, v_loss, pri_loss) = train_step_active(
                params, opt_state, curr_state, b_imgs, b_tasks, b_lbls, b_masks, k_batch, current_aux_weight
            )
            mode = "Active"
            print_stats = f"Loss={loss:.4f} (Cls={c_loss:.4f}, Pol={p_loss:.4f}, Val={v_loss:.4f}, AuxW={current_aux_weight:.2f}) | Acc={acc:.4f}"
            
        if i % 500 == 0:
             print(f"Step {i:04d} [{mode}]: {print_stats}")
             
    # Save Params
    print("Saving parameters...")
    with open("visual_search_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Saved to visual_search_params.pkl")

if __name__ == "__main__":
    train_visual_search()
