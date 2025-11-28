import jax
import jax.numpy as np
from typing import NamedTuple, Optional, Tuple

# Phase 1: Data Structures & Initialization

class Hyperparameters(NamedTuple):
    n_regions: int = 84
    n_heads: int = 8
    d_k: int = 16
    d_v: int = 16
    d_model: int = 16
    lam: float = 0.9 # forgetting factor lambda

class NetworkState(NamedTuple):
    M: jax.Array # (B, N, H, Dv, Dk)

class CTMHSAParams(NamedTuple):
    W_Q: jax.Array # (H, d_model, d_k)
    W_K: jax.Array # (H, d_model, d_k)
    W_V: jax.Array # (H, d_model, d_v)
    W_Y: jax.Array # (H, d_v, d_model)
    C: jax.Array   # (N, N)

def init_ct_mhsa(hp: Hyperparameters, key, batch_size: int = 1, initial_c: Optional[jax.Array] = None) -> Tuple[CTMHSAParams, NetworkState]:
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

    params = CTMHSAParams(W_Q=W_Q, W_K=W_K, W_V=W_V, W_Y=W_Y, C=C)
    
    # Initialize State M
    # Shape: (B, N, H, Dv, Dk)
    M_shape = (batch_size, hp.n_regions, hp.n_heads, hp.d_v, hp.d_k)
    M = np.zeros(M_shape)
    
    state = NetworkState(M=M)
    
    return params, state

# Phase 2: The Cortical Microcircuit (The "Head")

def compute_projections(params: CTMHSAParams, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute q, k, v projections.
    x: (B, N, d_model)
    Returns: q, k, v with shape (B, N, H, D_k/v)
    """
    # x: (B, N, Dm)
    # W: (H, Dm, Dk/v)
    # Out: (B, N, H, Dk/v)
    q = np.einsum('bnd,hdk->bnhk', x, params.W_Q)
    k = np.einsum('bnd,hdk->bnhk', x, params.W_K)
    v = np.einsum('bnd,hdv->bnhv', x, params.W_V)
    
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

def update_memory_l23(state: NetworkState, k: jax.Array, v: jax.Array, hp: Hyperparameters) -> NetworkState:
    """
    Update memory M using Delta Rule.
    M_t = M_{t-1} + lambda * (v_t * k_t^T - M_{t-1})
        = (1 - lambda) * M_{t-1} + lambda * (v_t * k_t^T)
    """
    # k: (B, N, H, Dk)
    # v: (B, N, H, Dv)
    # M: (B, N, H, Dv, Dk)
    
    # Outer product v * k^T -> (B, N, H, Dv, Dk)
    # Einsum: bnhv, bnhk -> bnhvk
    update = np.einsum('bnhv,bnhk->bnhvk', v, k)
    
    # Apply decay
    # M_new = (1 - hp.lam) * M + hp.lam * update
    M_new = (1 - hp.lam) * state.M + hp.lam * update
    
    return NetworkState(M=M_new)

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

def mhsa_step(params: CTMHSAParams, state: NetworkState, x: jax.Array, hp: Hyperparameters) -> Tuple[NetworkState, jax.Array]:
    """
    Single time step of MHSA block.
    """
    q, k, v = compute_projections(params, x)
    
    # Update Memory (L2/3)
    # Note: "Recurrent key-value memory (L2/3)" implies we update THEN read or READ then update?
    # Plan Phase 2: 
    # 1. Calc q, k, v.
    # 2. Update Memory.
    # 3. Retrieve Query.
    # This order (Update -> Retrieve) is typical for "Linear Transformers" (RNN view).
    state_new = update_memory_l23(state, k, v, hp)
    
    # Retrieve (L5)
    o = retrieve_query_l5(state_new, q)
    
    # Aggregate
    y = aggregate_heads(o, params)
    
    return state_new, y

# Phase 3: The Connectome & Time Loop

def network_coupling(y_t: jax.Array, params: CTMHSAParams, external_input: jax.Array) -> jax.Array:
    """
    Compute next input x_{t+1} based on previous output y_t and connectivity C.
    x_{t+1, i} = sum_j C_{ij} y_{t, j} + External_i
    """
    # y_t: (B, N, D_model)
    # C: (N, N)
    # External: (B, N, D_model)
    
    # C dot y -> (B, N, D)
    # Einsum: ij, bjd -> bid
    coupled_input = np.einsum('ij,bjd->bid', params.C, y_t)
    
    return coupled_input + external_input

def scan_ct_mhsa(params: CTMHSAParams, init_state: NetworkState, inputs: jax.Array, hp: Hyperparameters) -> Tuple[Tuple[NetworkState, jax.Array], jax.Array]:
    """
    Run the CT-MHSA model over a sequence of inputs.
    inputs: (T, B, N, D_model) - External inputs over time.
    Returns:
        (final_state, final_y), outputs_sequence
    """
    
    # Initial carry: (State, y_prev)
    # We assume y_prev starts at zero.
    B = inputs.shape[1]
    y_init = np.zeros((B, hp.n_regions, hp.d_model))
    carry_init = (init_state, y_init)
    
    def step_fn(carry, input_t):
        state_prev, y_prev = carry
        
        # Coupling
        x_t = network_coupling(y_prev, params, input_t)
        
        # Microcircuit Step
        state_new, y_new = mhsa_step(params, state_prev, x_t, hp)
        
        new_carry = (state_new, y_new)
        return new_carry, y_new

    final_carry, outputs = jax.lax.scan(step_fn, carry_init, inputs)
    
    return final_carry, outputs