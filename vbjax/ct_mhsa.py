import jax
import jax.numpy as np
from typing import NamedTuple, Optional, Tuple
from .coupling import make_delay_helper

# Phase 1: Data Structures & Initialization


class Hyperparameters(NamedTuple):
    n_regions: int = 84
    n_heads: int = 8
    d_k: int = 16
    d_v: int = 16
    d_model: int = 16
    lam: float = 0.9  # forgetting factor lambda
    dt: float = 0.1
    v_c: float = 10.0
    steps_per_token: int = 1


class NetworkState(NamedTuple):
    M: jax.Array  # (B, N, H, Dv, Dk)
    history: Optional[jax.Array] = None  # (L, B, N, D)
    step: int = 0


class CTMHSAParams(NamedTuple):
    W_Q: jax.Array  # (H, d_model, d_k)
    W_K: jax.Array  # (H, d_model, d_k)
    W_V: jax.Array  # (H, d_model, d_v)
    W_Y: jax.Array  # (H, d_v, d_model)
    C: jax.Array  # (N, N)
    ln_gamma: Optional[jax.Array] = None  # (d_model,)
    ln_beta: Optional[jax.Array] = None  # (d_model,)


def layer_norm(
    x: jax.Array, gamma: jax.Array, beta: jax.Array, eps: float = 1e-5
) -> jax.Array:
    """Layer Normalization along the last dimension."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def init_ct_mhsa(
    hp: Hyperparameters,
    key,
    batch_size: int = 1,
    initial_c: Optional[jax.Array] = None,
    lengths: Optional[jax.Array] = None,
) -> Tuple[CTMHSAParams, NetworkState, Optional[jax.Array]]:
    """Initialize parameters and state for Cortico-Thalamic MHSA."""

    k_q, k_k, k_v, k_y, k_c = jax.random.split(key, 5)

    def init_linear(key, shape):
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (shape[1] + shape[2]))
        return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

    W_Q = init_linear(k_q, (hp.n_heads, hp.d_model, hp.d_k))
    W_K = init_linear(k_k, (hp.n_heads, hp.d_model, hp.d_k))
    W_V = init_linear(k_v, (hp.n_heads, hp.d_model, hp.d_v))
    W_Y = init_linear(k_y, (hp.n_heads, hp.d_v, hp.d_model))

    if initial_c is None:
        # Initialize with identity or random small values if not provided?
        # Plan says "Initialize Structural Connectivity C (fixed or learnable mask)".
        # We'll use random normal scaled down for now if not provided.
        C = jax.random.normal(k_c, (hp.n_regions, hp.n_regions)) * 0.1
    else:
        C = initial_c

    lags = None
    history = None
    if lengths is not None:
        # Dummy weights for helper
        dummy_w = np.zeros((hp.n_regions, hp.n_regions))
        dh = make_delay_helper(dummy_w, lengths, hp.dt, hp.v_c)
        lags = dh.lags
        # History buffer: (max_lag, B, N, D)
        # Ensure max_lag is at least 1
        history = np.zeros((dh.max_lag, batch_size, hp.n_regions, hp.d_model))

    ln_gamma = np.ones((hp.d_model,))
    ln_beta = np.zeros((hp.d_model,))

    params = CTMHSAParams(
        W_Q=W_Q, W_K=W_K, W_V=W_V, W_Y=W_Y, C=C, ln_gamma=ln_gamma, ln_beta=ln_beta
    )

    # Initialize State M
    # Shape: (B, N, H, Dv, Dk)
    M_shape = (batch_size, hp.n_regions, hp.n_heads, hp.d_v, hp.d_k)
    M = np.zeros(M_shape)

    state = NetworkState(M=M, history=history, step=0)

    return params, state, lags


# Phase 2: The Cortical Microcircuit (The "Head")


def compute_projections(
    params: CTMHSAParams, x: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute q, k, v projections.
    x: (B, N, d_model)
    Returns: q, k, v with shape (B, N, H, D_k/v)
    """
    # x: (B, N, Dm)
    # W: (H, Dm, Dk/v)
    # Out: (B, N, H, Dk/v)
    q = np.einsum("bnd,hdk->bnhk", x, params.W_Q)
    k = np.einsum("bnd,hdk->bnhk", x, params.W_K)
    v = np.einsum("bnd,hdv->bnhv", x, params.W_V)

    # Activation phi (identity or ReLU). Plan mentions "Apply activation phi ... to k and q".
    # We'll use identity for now as simple linear attention often does, or elu+1.
    # "Apply activation phi (e.g., identity or ReLU)"
    # Let's use identity as default or maybe a simple relu.
    # For "Fast Weights" / Linear Transformers, often use elu()+1 or relu.
    # I'll stick to identity for now to match "linear projections" strict sense,
    # but usually strictly positive keys/queries help stability.
    # Let's use ReLU to be safe and follow common linear attention practices if not specified strictly.
    # Plan: "Apply activation phi (e.g., identity or ReLU) to k and q"
    # I will use ReLU.
    k = jax.nn.relu(k)
    q = jax.nn.relu(q)

    return q, k, v


def update_memory_l23(state: NetworkState, k: jax.Array, v: jax.Array, hp: Hyperparameters) -> Tuple[NetworkState, jax.Array]:
    """
    Update memory M using Delta Rule.
    M_t = M_{t-1} + lambda * (v_t * k_t^T - M_{t-1})
    Returns state and surprise (Frobenius norm of delta M).
    """
    # k: (B, N, H, Dk)
    # v: (B, N, H, Dv)
    # M: (B, N, H, Dv, Dk)
    
    # Outer product v * k^T -> (B, N, H, Dv, Dk)
    # Einsum: bnhv, bnhk -> bnhvk
    target = np.einsum('bnhv,bnhk->bnhvk', v, k)
    
    # Delta M = lambda * (target - M_prev)
    delta_M = hp.lam * (target - state.M)
    M_new = state.M + delta_M
    
    # Surprise: Frobenius norm over last two dims (Dv, Dk)
    # shape: (B, N, H)
    surprise = np.linalg.norm(delta_M, axis=(-2, -1))
    
    return state._replace(M=M_new), surprise

def retrieve_query_l5(state: NetworkState, q: jax.Array) -> jax.Array:
    """
    Retrieve output from memory.
    o_t = M_t * q_t
    """
    # M: (B, N, H, Dv, Dk)
    # q: (B, N, H, Dk)
    # Out: (B, N, H, Dv)
    # Einsum: bnhvk, bnhk -> bnhv
    o = np.einsum('bnhvk,bnhk->bnhv', state.M, q)
    return o

def aggregate_heads(o: jax.Array, params: CTMHSAParams) -> jax.Array:
    """
    Aggregate head outputs.
    y_t = sum_h W_Y^h o_t^h
    """
    # o: (B, N, H, Dv)
    # W_Y: (H, Dv, D_model)
    # Out: (B, N, D_model)
    # Einsum: bnhv, hvd -> bnd
    y = np.einsum('bnhv,hvd->bnd', o, params.W_Y)
    return y

def mhsa_step(params: CTMHSAParams, state: NetworkState, x: jax.Array, hp: Hyperparameters) -> Tuple[NetworkState, jax.Array, jax.Array]:
    """
    Single time step of MHSA block.
    Returns: state, y, surprise
    """
    q, k, v = compute_projections(params, x)
    
    # Update Memory (L2/3)
    state_new, surprise = update_memory_l23(state, k, v, hp)
    
    # Retrieve (L5)
    o = retrieve_query_l5(state_new, q)
    
    # Aggregate
    y = aggregate_heads(o, params)
    
    # Residual Connection
    y = y + x
    
    # Layer Norm
    if params.ln_gamma is not None and params.ln_beta is not None:
        y = layer_norm(y, params.ln_gamma, params.ln_beta)
    
    return state_new, y, surprise



# Phase 3: The Connectome & Time Loop


def network_coupling(
    y_t: jax.Array,
    params: CTMHSAParams,
    external_input: jax.Array,
    history: Optional[jax.Array] = None,
    step: int = 0,
    lags: Optional[jax.Array] = None,
) -> jax.Array:
    """
    Compute next input x_{t+1} based on previous output y_t and connectivity C.
    x_{t+1, i} = sum_j C_{ij} y_{t, j} + External_i
    """

    if history is not None and lags is not None:
        # Delayed coupling
        L_max = history.shape[0]
        indices = (step - lags) % L_max  # (N, N)

        def apply_delay_batch(hist_b):
            # hist_b: (L, N, D)
            # Gather delayed values
            col_indices = np.tile(np.arange(lags.shape[1]), (lags.shape[0], 1))
            y_delayed = hist_b[indices, col_indices, :]  # (N, N, D)
            return np.einsum("ij,ijd->id", params.C, y_delayed)

        # history: (L, B, N, D) -> transpose to (B, L, N, D) for vmap
        coupled_input = jax.vmap(apply_delay_batch)(history.transpose(1, 0, 2, 3))

    else:
        # y_t: (B, N, D_model)
        # C: (N, N)
        # External: (B, N, D_model)

        # C dot y -> (B, N, D)
        # Einsum: ij, bjd -> bid
        coupled_input = np.einsum("ij,bjd->bid", params.C, y_t)

    return coupled_input + external_input


def scan_ct_mhsa(params: CTMHSAParams, init_state: NetworkState, inputs: jax.Array, hp: Hyperparameters, lags: Optional[jax.Array] = None) -> Tuple[Tuple[NetworkState, jax.Array], Tuple[jax.Array, jax.Array]]:
    """
    Run the CT-MHSA model over a sequence of inputs.
    inputs: (T, B, N, D_model) - External inputs over time.
    Returns:
        (final_state, final_y), (outputs_sequence, surprise_trace)
        outputs_sequence: (T, B, N, D)
        surprise_trace: (T, K, B, N, H)
    """
    
    # Initial carry: (State, y_prev)
    # We assume y_prev starts at zero.
    B = inputs.shape[1]
    y_init = np.zeros((B, hp.n_regions, hp.d_model))
    carry_init = (init_state, y_init)
    
    def step_fn(carry, input_t):
        
        def micro_step_fn(inner_carry, _):
            state_prev, y_prev = inner_carry
            
            history = state_prev.history
            step = state_prev.step
            
            # Update history if present
            if history is not None:
                L_max = history.shape[0]
                # Write y_prev (which is y_{t-1}) to (step - 1) % L
                # For step=0, we write y_init to L-1.
                idx = (step - 1) % L_max
                history = history.at[idx].set(y_prev)
            
            # Coupling
            x_t = network_coupling(y_prev, params, input_t, history=history, step=step, lags=lags)
            
            # Microcircuit Step
            state_new, y_new, surprise = mhsa_step(params, state_prev, x_t, hp)
            
            # Update state with new history and increment step
            state_new = state_new._replace(history=history, step=step + 1)
            
            return (state_new, y_new), surprise

        # Run micro-steps using scan to trace surprise
        final_inner_carry, surprise_trace = jax.lax.scan(micro_step_fn, carry, None, length=hp.steps_per_token)
        
        # Return result of last micro step as the output for this sequence step
        # Output structure: (final_y, surprise_trace)
        return final_inner_carry, (final_inner_carry[1], surprise_trace)

    final_carry, outputs = jax.lax.scan(step_fn, carry_init, inputs)
    
    return final_carry, outputs
