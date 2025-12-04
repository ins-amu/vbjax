import jax
import jax.numpy as np
from typing import NamedTuple, Tuple, Optional
from vbjax.ct_mhsa import CTMHSAParams, NetworkState, Hyperparameters, mhsa_step, init_ct_mhsa, network_coupling

class VisualSearchHyperparameters(NamedTuple):
    mhsa: Hyperparameters
    patch_size: int = 16
    n_tasks: int = 2
    n_classes: int = 3 # R, G, B (max classes)
    retina_channels: Tuple[int, ...] = (8, 16)
    max_steps: int = 100 # Max horizon for time embeddings

class VisualSearchParams(NamedTuple):
    core: CTMHSAParams
    # Retina
    conv1_w: jax.Array
    conv1_b: jax.Array
    conv2_w: jax.Array
    conv2_b: jax.Array
    retina_out_w: jax.Array # flattened -> d_model
    retina_out_b: jax.Array
    # Embeddings
    pos_embed_w: jax.Array
    pos_embed_b: jax.Array
    task_embed: jax.Array # (n_tasks, d_model)
    time_embed: jax.Array # (max_steps, d_model)
    # Heads
    head_answer_w: jax.Array
    head_answer_b: jax.Array
    head_saccade_w: jax.Array
    head_saccade_b: jax.Array
    head_value_w: jax.Array
    head_value_b: jax.Array
    head_priority_w: jax.Array # New: for PCIP supervision
    head_priority_b: jax.Array

# Right Hemisphere Indices (0-37)
IDX_R_FEF = 7
IDX_R_PCIP = 14 # rPCIP (Intraparietal / Priority Map)
IDX_R_PFC = 18  # rPFCDL (Dorsolateral PFC)
IDX_R_V1 = 35   # rV1 (Primary Visual)

def init_visual_search(
    hp: VisualSearchHyperparameters, 
    key,
    connectome_weights: Optional[jax.Array] = None,
    connectome_lengths: Optional[jax.Array] = None
) -> Tuple[VisualSearchParams, NetworkState]:
    k_core, k_c1, k_c2, k_ro, k_pe, k_te, k_ha, k_hs, k_hv, k_hp, k_time = jax.random.split(key, 11)
    
    # Init Core
    # Pass the real connectome if provided
    
    # FORCE CONNECTIVITY FIX for Visual Search
    # Ensure signal propagation from Visual Input (V1) to Decision (PFC) and Action (FEF)
    if connectome_weights is not None:
        # V1 (35) -> PFC (18) (Ventral/Dorsal Stream shortcut)
        connectome_weights = connectome_weights.at[IDX_R_PFC, IDX_R_V1].set(0.5) 
        # PFC (18) -> V1 (35) (Top-down attention)
        connectome_weights = connectome_weights.at[IDX_R_V1, IDX_R_PFC].set(0.5) 
        
        # V1 (35) -> FEF (7) (Saliency map shortcut)
        connectome_weights = connectome_weights.at[IDX_R_FEF, IDX_R_V1].set(0.5)
        
        # PFC (18) -> FEF (7) (Executive control of eye)
        connectome_weights = connectome_weights.at[IDX_R_FEF, IDX_R_PFC].set(0.5)

    core_params, core_state = init_ct_mhsa(
        hp.mhsa, 
        key=k_core, 
        batch_size=1,
        initial_c=connectome_weights,
        lengths=connectome_lengths
    ) # Batch size handled dynamically in vmap
    
    def init_dense(k, n_in, n_out):
        lim = np.sqrt(6 / (n_in + n_out))
        return jax.random.uniform(k, (n_in, n_out), minval=-lim, maxval=lim), np.zeros((n_out,))
    
    def init_conv(k, c_in, c_out, k_size=3):
        # HWIO layout for flax/jax.lax.conv usually, but let's check
        # jax.lax.conv_general_dilated expects lhs (N, H, W, C), rhs (K, K, I, O) if dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        shape = (k_size, k_size, c_in, c_out)
        lim = np.sqrt(6 / (c_in*k_size*k_size + c_out))
        return jax.random.uniform(k, shape, minval=-lim, maxval=lim), np.zeros((c_out,))

    # Retina
    # Input: 16x16x3
    # C1: 3 -> 8
    c1_w, c1_b = init_conv(k_c1, 3, hp.retina_channels[0])
    # C2: 8 -> 16
    c2_w, c2_b = init_conv(k_c2, hp.retina_channels[0], hp.retina_channels[1])
    
    # Spatial Softmax Flatten Size
    # Output is (B, C*2) because we get x,y for each channel
    flat_size = hp.retina_channels[1] * 2
    
    ro_w, ro_b = init_dense(k_ro, flat_size, hp.mhsa.d_model)
    
    # Embeddings
    pe_w, pe_b = init_dense(k_pe, 2, hp.mhsa.d_model)
    task_embed = jax.random.normal(k_te, (hp.n_tasks, hp.mhsa.d_model)) * 0.1
    time_embed = jax.random.normal(k_time, (hp.max_steps, hp.mhsa.d_model)) * 0.1
    
    # Heads
    # Output of core is (B, N, d_model)
    # We pool over N (regions) or use a specific readout?
    # The paper/plan says "y_t -> Class". y_t is (B, N, D).
    # We'll assume we average over regions for the "global" answer, or use a specific readout node.
    # Let's Average Pooling for now: (B, D).
    ha_w, ha_b = init_dense(k_ha, hp.mhsa.d_model, hp.n_classes)
    hs_w, hs_b = init_dense(k_hs, hp.mhsa.d_model, 2) # dx, dy
    hv_w, hv_b = init_dense(k_hv, hp.mhsa.d_model, 1) # Value (Scalar)
    hp_w, hp_b = init_dense(k_hp, hp.mhsa.d_model, 1) # Priority (Scalar)
    
    params = VisualSearchParams(
        core=core_params,
        conv1_w=c1_w, conv1_b=c1_b,
        conv2_w=c2_w, conv2_b=c2_b,
        retina_out_w=ro_w, retina_out_b=ro_b,
        pos_embed_w=pe_w, pos_embed_b=pe_b,
        task_embed=task_embed,
        time_embed=time_embed,
        head_answer_w=ha_w, head_answer_b=ha_b,
        head_saccade_w=hs_w, head_saccade_b=hs_b,
        head_value_w=hv_w, head_value_b=hv_b,
        head_priority_w=hp_w, head_priority_b=hp_b
    )
    
    return params, core_state

def spatial_softmax(features):
    """
    features: (Batch, Height, Width, Channels)
    Returns: (Batch, Channels * 2) -> flattened list of x,y coords
    """
    B, H, W, C = features.shape
    
    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    # Normalize to [-1, 1]
    y_grid = (y_grid / (H - 1 + 1e-6) * 2 - 1).astype(np.float32)
    x_grid = (x_grid / (W - 1 + 1e-6) * 2 - 1).astype(np.float32)
    
    # Flatten spatial dims: (B, H*W, C)
    features_flat = features.reshape(B, H * W, C)
    
    # Softmax over space to get attention maps (spatial dimension is axis 1)
    # We want softmax over H*W for each channel independently
    probs = jax.nn.softmax(features_flat, axis=1) # (B, H*W, C)
    
    # Compute expected X and Y for each channel
    # Sum(Prob * Coord)
    # x_grid flat: (H*W,) -> (1, H*W, 1) broadcast
    expected_x = np.sum(probs * x_grid.reshape(1, -1, 1), axis=1) # (B, C)
    expected_y = np.sum(probs * y_grid.reshape(1, -1, 1), axis=1) # (B, C)
    
    # Concatenate (B, C*2)
    return np.concatenate([expected_x, expected_y], axis=-1)

def retina_forward(params: VisualSearchParams, patch: jax.Array) -> jax.Array:
    # patch: (B, 16, 16, 3)
    # Conv1
    # w: (3, 3, 3, 8)
    x = jax.lax.conv_general_dilated(
        patch, params.conv1_w, 
        window_strides=(2, 2),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = jax.nn.relu(x + params.conv1_b)
    
    # Conv2
    x = jax.lax.conv_general_dilated(
        x, params.conv2_w, 
        window_strides=(2, 2),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = jax.nn.relu(x + params.conv2_b)
    
    # Spatial Softmax instead of Flatten
    x = spatial_softmax(x)
    
    # Projection
    x = x @ params.retina_out_w + params.retina_out_b
    return x

def agent_step(
    params: VisualSearchParams,
    state: NetworkState,
    patch: jax.Array,
    pos: jax.Array,
    task_idx: jax.Array,
    step_idx: int,
    hp: VisualSearchHyperparameters
) -> Tuple[NetworkState, Tuple[jax.Array, jax.Array]]:
    """
    Processes one visual fixation (saccade step).
    Runs the core micro-circuit for n_micro_steps.
    """
    
    # 1. Encode Inputs
    vis_feat = retina_forward(params, patch) # (B, D)
    pos_feat = pos @ params.pos_embed_w + params.pos_embed_b # (B, D)
    task_feat = params.task_embed[task_idx] # (B, D)
    time_feat = params.time_embed[step_idx] # (B, D) (Assuming step_idx is valid index)
    
    # Fuse: Sparse Injection
    # Core input x needs to be (B, N, D).
    
    # Initialize with zeros
    B = patch.shape[0]
    N = params.core.C.shape[0]
    core_input = np.zeros((B, N, hp.mhsa.d_model))
    
    # Inject Visual (V1) + Position (V1/Dorsal stream)
    # V1 acts as the entry point
    core_input = core_input.at[:, IDX_R_V1, :].set(vis_feat + pos_feat)
    
    # Inject Task Context (PFC)
    # PFC holds the search goal
    core_input = core_input.at[:, IDX_R_PFC, :].set(task_feat)
    
    # 2. Micro-steps
    def loop_body(carry, _):
        curr_state, y_prev = carry
        
        # Network Coupling (Transmission)
        # x_t = C * y_{t-1} + External Input
        x_t = network_coupling(
            y_prev, params.core, core_input, 
            history=curr_state.history, 
            step=curr_state.step, 
            lags=curr_state.lags
        )
        
        # Microcircuit Step
        # Note: mhsa_step uses x_t to update M and produce y_t
        # It returns state with updated M, but NOT updated history/step
        new_state_m, y, surprise = mhsa_step(params.core, curr_state, x_t, hp.mhsa)
        
        # Update History & Step
        history = curr_state.history
        step = curr_state.step
        
        if history is not None:
            L_max = history.shape[0]
            # Write y_prev (output of previous step) to history
            # For step=0, we write y_init (zeros) to history
            idx = (step - 1) % L_max
            history = history.at[idx].set(y_prev)
            
        final_state = new_state_m._replace(history=history, step=step + 1)

        return (final_state, y), surprise
    
    # Initial y (dummy)
    y_init = np.zeros((B, N, hp.mhsa.d_model))
    
    # We'll use lax.scan to capture surprise trace
    (final_state, final_y), surprise_trace = jax.lax.scan(loop_body, (state, y_init), None, length=hp.mhsa.steps_per_token)
    
    # surprise_trace: (K, B, N, H) -> (B, K, N, H)
    surprise_trace = np.transpose(surprise_trace, (1, 0, 2, 3))
    
    # 3. Heads
    # final_y: (B, N, D)
    
    # Saccade from FEF (Actor)
    fef_activity = final_y[:, IDX_R_FEF, :]
    saccade = fef_activity @ params.head_saccade_w + params.head_saccade_b
    # Tanh for saccade to keep in [-1, 1]? Or raw?
    # Paper/Plan usually implies continuous control. Let's use Tanh to bound it comfortably.
    saccade = np.tanh(saccade) 
    
    # Classification / Value from PFC (Decision/Critic)
    pfc_activity = final_y[:, IDX_R_PFC, :]
    
    logits = pfc_activity @ params.head_answer_w + params.head_answer_b
    
    value = pfc_activity @ params.head_value_w + params.head_value_b
    value = value.squeeze(-1) # (B,)
    
    # Priority from PCIP (Priority Map Supervision)
    pcip_activity = final_y[:, IDX_R_PCIP, :]
    priority = pcip_activity @ params.head_priority_w + params.head_priority_b
    priority = priority.squeeze(-1) # (B,)
    
    return final_state, (logits, saccade, value, surprise_trace, priority)

def agent_step_eval(
    params: VisualSearchParams,
    state: NetworkState,
    patch: jax.Array,
    pos: jax.Array,
    task_idx: jax.Array,
    step_idx: int,
    hp: VisualSearchHyperparameters
) -> Tuple[NetworkState, Tuple[jax.Array, jax.Array]]:
    """
    Processes one visual fixation (saccade step) deterministically (no exploration noise).
    Runs the core micro-circuit for n_micro_steps.
    """
    
    # 1. Encode Inputs
    vis_feat = retina_forward(params, patch) # (B, D)
    pos_feat = pos @ params.pos_embed_w + params.pos_embed_b # (B, D)
    task_feat = params.task_embed[task_idx] # (B, D)
    time_feat = params.time_embed[step_idx] # (B, D) (Assuming step_idx is valid index)
    
    # Fuse: Sparse Injection
    # Core input x needs to be (B, N, D).
    
    # Initialize with zeros
    B = patch.shape[0]
    N = params.core.C.shape[0]
    core_input = np.zeros((B, N, hp.mhsa.d_model))
    
    # Inject Visual (V1) + Position (V1/Dorsal stream)
    # V1 acts as the entry point
    core_input = core_input.at[:, IDX_R_V1, :].set(vis_feat + pos_feat)
    
    # Inject Task Context (PFC)
    # PFC holds the search goal
    core_input = core_input.at[:, IDX_R_PFC, :].set(task_feat)
    
    # 2. Micro-steps
    def loop_body(carry, _):
        curr_state, y_prev = carry
        
        # Network Coupling (Transmission)
        # x_t = C * y_{t-1} + External Input
        x_t = network_coupling(
            y_prev, params.core, core_input, 
            history=curr_state.history, 
            step=curr_state.step, 
            lags=curr_state.lags
        )
        
        # Microcircuit Step
        # Note: mhsa_step uses x_t to update M and produce y_t
        # It returns state with updated M, but NOT updated history/step
        new_state_m, y, surprise = mhsa_step(params.core, curr_state, x_t, hp.mhsa)
        
        # Update History & Step
        history = curr_state.history
        step = curr_state.step
        
        if history is not None:
            L_max = history.shape[0]
            # Write y_prev (output of previous step) to history
            # For step=0, we write y_init (zeros) to history
            idx = (step - 1) % L_max
            history = history.at[idx].set(y_prev)
            
        final_state = new_state_m._replace(history=history, step=step + 1)

        return (final_state, y), surprise
    
    # Initial y (dummy)
    y_init = np.zeros((B, N, hp.mhsa.d_model))
    
    # We'll use lax.scan to capture surprise trace
    (final_state, final_y), surprise_trace = jax.lax.scan(loop_body, (state, y_init), None, length=hp.mhsa.steps_per_token)
    
    # surprise_trace: (K, B, N, H) -> (B, K, N, H)
    surprise_trace = np.transpose(surprise_trace, (1, 0, 2, 3))
    
    # 3. Heads
    # final_y: (B, N, D)
    
    # Saccade from FEF (Actor)
    fef_activity = final_y[:, IDX_R_FEF, :]
    saccade = fef_activity @ params.head_saccade_w + params.head_saccade_b
    saccade = np.tanh(saccade) 
    
    # Classification / Value from PFC (Decision/Critic)
    pfc_activity = final_y[:, IDX_R_PFC, :]
    
    logits = pfc_activity @ params.head_answer_w + params.head_answer_b
    
    value = pfc_activity @ params.head_value_w + params.head_value_b
    value = value.squeeze(-1) # (B,)
    
    # Priority from PCIP (Priority Map Supervision)
    pcip_activity = final_y[:, IDX_R_PCIP, :]
    priority = pcip_activity @ params.head_priority_w + params.head_priority_b
    priority = priority.squeeze(-1) # (B,)
    
    return final_state, (logits, saccade, value, surprise_trace, priority)
