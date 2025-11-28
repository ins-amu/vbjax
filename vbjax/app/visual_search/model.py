import jax
import jax.numpy as np
from typing import NamedTuple, Tuple, Optional
from vbjax.ct_mhsa import CTMHSAParams, NetworkState, Hyperparameters, mhsa_step, init_ct_mhsa

class VisualSearchHyperparameters(NamedTuple):
    mhsa: Hyperparameters
    patch_size: int = 16
    n_micro_steps: int = 5
    n_tasks: int = 2
    n_classes: int = 3 # R, G, B (max classes)
    retina_channels: Tuple[int, ...] = (8, 16)

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
    # Heads
    head_answer_w: jax.Array
    head_answer_b: jax.Array
    head_saccade_w: jax.Array
    head_saccade_b: jax.Array

def init_visual_search(hp: VisualSearchHyperparameters, key) -> Tuple[VisualSearchParams, NetworkState]:
    k_core, k_c1, k_c2, k_ro, k_pe, k_te, k_ha, k_hs = jax.random.split(key, 8)
    
    # Init Core
    core_params, core_state, _ = init_ct_mhsa(hp.mhsa, key=k_core, batch_size=1) # Batch size handled dynamically in vmap
    
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
    
    # Flatten size: 16x16 -> (stride 2) -> 8x8 -> (stride 2) -> 4x4
    # 4 * 4 * 16 = 256
    flat_size = 4 * 4 * hp.retina_channels[1]
    ro_w, ro_b = init_dense(k_ro, flat_size, hp.mhsa.d_model)
    
    # Embeddings
    pe_w, pe_b = init_dense(k_pe, 2, hp.mhsa.d_model)
    task_embed = jax.random.normal(k_te, (hp.n_tasks, hp.mhsa.d_model)) * 0.1
    
    # Heads
    # Output of core is (B, N, d_model)
    # We pool over N (regions) or use a specific readout?
    # The paper/plan says "y_t -> Class". y_t is (B, N, D).
    # We'll assume we average over regions for the "global" answer, or use a specific readout node.
    # Let's Average Pooling for now: (B, D).
    ha_w, ha_b = init_dense(k_ha, hp.mhsa.d_model, hp.n_classes)
    hs_w, hs_b = init_dense(k_hs, hp.mhsa.d_model, 2) # dx, dy
    
    params = VisualSearchParams(
        core=core_params,
        conv1_w=c1_w, conv1_b=c1_b,
        conv2_w=c2_w, conv2_b=c2_b,
        retina_out_w=ro_w, retina_out_b=ro_b,
        pos_embed_w=pe_w, pos_embed_b=pe_b,
        task_embed=task_embed,
        head_answer_w=ha_w, head_answer_b=ha_b,
        head_saccade_w=hs_w, head_saccade_b=hs_b
    )
    
    return params, core_state

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
    
    # Flatten
    B = patch.shape[0]
    x = x.reshape((B, -1))
    
    # Projection
    x = x @ params.retina_out_w + params.retina_out_b
    return x

def agent_step(
    params: VisualSearchParams,
    state: NetworkState,
    patch: jax.Array,
    pos: jax.Array,
    task_idx: jax.Array,
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
    
    # Fuse: Sum
    # Core input x needs to be (B, N, D).
    # Our features are (B, D). We broadcast to all regions?
    # Or we only drive specific regions (V1)?
    # "Input a one-hot... as well as the 16x16 square".
    # Let's broadcast to all regions for simplicity first, effectively "global context".
    combined_feat = vis_feat + pos_feat + task_feat
    
    # Expand to (B, N, D)
    N = params.core.C.shape[0]
    core_input = np.tile(combined_feat[:, None, :], (1, N, 1))
    
    # 2. Micro-steps
    def loop_body(i, carry):
        curr_state, _ = carry
        # We feed the same static sensory input at each microstep?
        # Yes, typical for "settling" dynamics.
        new_state, y, _ = mhsa_step(params.core, curr_state, core_input, hp.mhsa)
        return (new_state, y)
    
    # Initial y (dummy)
    y_init = np.zeros_like(core_input)
    
    # We'll use lax.fori_loop
    final_state, final_y = jax.lax.fori_loop(0, hp.n_micro_steps, loop_body, (state, y_init))
    
    # 3. Heads
    # final_y: (B, N, D)
    # Aggregate over N -> (B, D)
    y_agg = np.mean(final_y, axis=1)
    
    logits = y_agg @ params.head_answer_w + params.head_answer_b
    saccade = y_agg @ params.head_saccade_w + params.head_saccade_b
    # Tanh for saccade to keep in [-1, 1]? Or raw?
    # Paper/Plan usually implies continuous control. Let's use Tanh to bound it comfortably.
    saccade = np.tanh(saccade) 
    
    return final_state, (logits, saccade)

