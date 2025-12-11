import jax
import jax.numpy as np
from typing import NamedTuple, Optional, Tuple, Any

# --- Data Structures ---

class GLEHyperparameters(NamedTuple):
    d_model: int = 16
    d_k: int = 16
    d_v: int = 16
    n_heads: int = 4
    n_regions: int = 8
    dt: float = 0.1
    tau_m: float = 5.0
    tau_r: float = 5.0
    gamma: float = 0.5  # Nudging
    lam: float = 0.9    # Fast weight forgetting
    lr_w: float = 0.001 # Local learning rate

class GLEGradients(NamedTuple):
    # Accumulators for weight updates
    dW_Q: jax.Array
    dW_K: jax.Array
    dW_V: jax.Array
    dW_Y: jax.Array
    # dW_C: jax.Array # Connectome learning optional

class GLEState(NamedTuple):
    # Activity States (u)
    u_q: jax.Array
    u_k: jax.Array
    u_v: jax.Array
    u_y: jax.Array # Output
    
    # Error States (v / prosp_v)
    v_q: jax.Array
    v_k: jax.Array
    v_v: jax.Array
    v_y: jax.Array
    
    # Prospective States (for plasticity)
    prosp_v_q: jax.Array
    prosp_v_k: jax.Array
    prosp_v_v: jax.Array
    prosp_v_y: jax.Array
    
    # Core MHSA State (Fast Weights)
    M: jax.Array # (B, N, H, Dv, Dk)
    
    # Connectivity History
    history: Optional[jax.Array] = None
    step: int = 0

class CTMHSAParams(NamedTuple):
    W_Q: jax.Array  # (H, d_model, d_k)
    W_K: jax.Array  # (H, d_model, d_k)
    W_V: jax.Array  # (H, d_model, d_v)
    W_Y: jax.Array  # (H, d_v, d_model)
    C: jax.Array    # (N, N)

# --- Helper Functions ---

def phi(x):
    # Activation
    return np.tanh(x)

def phi_prime(x):
    # Derivative
    return 1.0 - np.tanh(x)**2

def gle_linear_dynamics(
    u: jax.Array, 
    v: jax.Array, 
    input_current: jax.Array, 
    error_current: jax.Array, 
    hp: GLEHyperparameters
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Computes one step of GLE dynamics for a generic neuron group.
    Returns: (u_next, v_next, prosp_v)
    """
    
    # Error Potential Dynamics
    # v_dot = (-v + error_current) / tau_r
    dv = (-v + error_current) / hp.tau_r
    v_next = v + hp.dt * dv
    prosp_v = v + hp.tau_m * dv # Cross-coupling for learning
    
    # Membrane Potential Dynamics
    # u_dot = (-u + input_current + gamma * prosp_v) / tau_m
    du = (-u + input_current + hp.gamma * prosp_v) / hp.tau_m
    u_next = u + hp.dt * du
    # prosp_u = u + hp.tau_r * du # For output
    
    return u_next, v_next, prosp_v

# --- Core Logic ---

def compute_projections_gle(
    params: CTMHSAParams, 
    x_in: jax.Array, # (B, N, D)
    state: GLEState,
    hp: GLEHyperparameters
) -> Tuple[GLEState, Tuple[jax.Array, jax.Array, jax.Array]]:
    """
    Computes Q, K, V activations using GLE dynamics.
    Instead of q = x @ W, we iterate u_q.
    """
    
    # 1. Forward Inputs (Currents)
    # W: (H, Dm, Dk)
    # x: (B, N, Dm)
    # Target: (B, N, H, Dk)
    
    # We broadcast x to heads for projection
    # Einsum: bnd, hdk -> bnhk
    curr_q = np.einsum("bnd, hdk -> bnhk", x_in, params.W_Q)
    curr_k = np.einsum("bnd, hdk -> bnhk", x_in, params.W_K)
    curr_v = np.einsum("bnd, hdv -> bnhv", x_in, params.W_V)
    
    # 2. Error Inputs (Feedback)
    # For Q, K, V, the error comes from the MHSA microcircuit logic (Attention gradient).
    # This is tricky without auto-diff.
    # In GLE, "Feedback" is usually v_post. But here post is the Attention mechanism.
    # We need to approximate the error arriving at Q, K, V from the future/output.
    #
    # Simplifying Assumption for GLE Phase 1:
    # We will assume error_current is 0 for Q/K/V for now, 
    # relying on the *recurrent* error injection at the Output layer to eventually 
    # propagate back via a separate mechanism or assume K/Q/V are fixed random projections 
    # and we only learn W_Y (Output).
    #
    # WAIT: The workplan says "Error Injection... v_pre <- W.T @ v_post".
    # We can't easily invert the Fast Weight "M" matrix analytically in one step.
    #
    # Strategy: 
    # For now, let's treat W_Q, W_K, W_V as standard GLE layers but with 
    # zero feedback error (unsupervised/random features) OR 
    # we implement a "Feedback Alignment" random matrix to project output error back to Q/K/V.
    # Let's use 0 error for now (Hebbian on input only?).
    # No, let's inject error from the Output Y back to V at least.
    #
    # For "V", the output "O" is roughly M @ Q.
    # If O has error, V should have error.
    # Let's start simple: Q, K, V dynamics driven only by Input.
    # The Learning will happen at W_Y (Output Projection).
    
    err_q = np.zeros_like(state.u_q)
    err_k = np.zeros_like(state.u_k)
    err_v = np.zeros_like(state.u_v)
    
    # 3. Dynamics
    # Q
    u_q, v_q, pv_q = gle_linear_dynamics(state.u_q, state.v_q, curr_q, err_q, hp)
    # K
    u_k, v_k, pv_k = gle_linear_dynamics(state.u_k, state.v_k, curr_k, err_k, hp)
    # V
    u_v, v_v, pv_v = gle_linear_dynamics(state.u_v, state.v_v, curr_v, err_v, hp)
    
    new_state = state._replace(
        u_q=u_q, v_q=v_q, prosp_v_q=pv_q,
        u_k=u_k, v_k=v_k, prosp_v_k=pv_k,
        u_v=u_v, v_v=v_v, prosp_v_v=pv_v
    )
    
    # Activations
    # Prosp_u = u + tau_r * du (Approximation)
    # We'll just use u_next for simplicity or recompute du.
    # Let's use phi(u_next)
    q = phi(u_q)
    k = phi(u_k)
    v = phi(u_v)
    
    return new_state, (q, k, v)

def update_fast_weights(
    M: jax.Array, 
    k: jax.Array, 
    v: jax.Array, 
    hp: GLEHyperparameters
) -> jax.Array:
    """
    M_t = M_{t-1} + lambda * (v * k^T - M_{t-1})
    """
    # k: (B, N, H, Dk)
    # v: (B, N, H, Dv)
    # Outer product: bnhv, bnhk -> bnhvk
    target = np.einsum('bnhv,bnhk->bnhvk', v, k)
    
    lam = hp.lam
    delta = lam * (target - M)
    return M + delta

def gle_mhsa_step(
    params: CTMHSAParams, 
    state: GLEState, 
    x_in: jax.Array, # (B, N, D)
    error_y_in: jax.Array, # (B, N, D) - Error injected at Y
    hp: GLEHyperparameters
) -> Tuple[GLEState, jax.Array, GLEGradients]:
    """
    One step of GLE-based MHSA.
    """
    
    # 1. Pre-calculate Projections (Stateless / Activation only)
    # We need Q, K, V activations to update M and compute O.
    # But we can't update U/V dynamics yet because we don't have error.
    # We use the *current* U state to derive activations for the forward pass.
    # q = phi(u_q), etc.
    q = phi(state.u_q)
    k = phi(state.u_k)
    v = phi(state.u_v)
    
    # 2. Fast Weight Update (M)
    M_new = update_fast_weights(state.M, k, v, hp)
    state = state._replace(M=M_new)
    
    # 3. Retrieve (Attention/Linear)
    # o = M @ q
    # bnhvk, bnhk -> bnhv
    o = np.einsum('bnhvk,bnhk->bnhv', state.M, q)
    
    # 4. Output Projection (Y)
    # Current: W_Y @ o
    curr_y = np.einsum('bnhv,hvd->bnd', o, params.W_Y)
    
    # 5. Output Dynamics (Y)
    u_y, v_y, pv_y = gle_linear_dynamics(state.u_y, state.v_y, curr_y, error_y_in, hp)
    state = state._replace(u_y=u_y, v_y=v_y, prosp_v_y=pv_y)
    
    r_y = phi(u_y)
    
    # --- FEEDBACK PATH (Backprop through structure via Transpose) ---
    
    # A. Backprop from Y to O (through W_Y)
    # pv_y: (B, N, Dm)
    # W_Y: (H, Dv, Dm)
    # err_o target: (B, N, H, Dv)
    # einsum: bnd, hvd -> bnhv
    err_o = np.einsum('bnd, hvd -> bnhv', pv_y, params.W_Y)
    
    # B. Backprop from O to Q (through M)
    # M: (B, N, H, Dv, Dk)
    # err_o: (B, N, H, Dv)
    # err_q target: (B, N, H, Dk)
    # Transpose M: swap Dv, Dk -> (..., Dk, Dv)
    # einsum: bnhvk, bnhv -> bnhk
    err_q = np.einsum('bnhvk, bnhv -> bnhk', state.M, err_o)
    
    # C. Backprop from O to V (through Q, approx)
    # If o = Mq = (M_prev + v k^T) q
    # do/dv = k^T q ... maybe too complex for simple GLE. 
    # Let's assume K/V feedback is random or zero for now, 
    # or just broadcast err_o back to V for "unspecific" error.
    # Let's rely on Q learning "where to look" primarily.
    err_k = np.zeros_like(state.u_k)
    err_v = np.zeros_like(state.u_v) 
    
    # --- RE-RUN PROJECTION DYNAMICS WITH ERROR ---
    # We computed projections earlier with 0 error. 
    # Now we have err_q. We need to update v_q properly.
    # Note: Ideally this happens simultaneously. 
    # In discrete time, we can just run the dynamics step for Q/K/V *here* using the error we just found,
    # instead of doing it at the top with zeros.
    
    # Let's move the dynamics logic here.
    
    # 1. Re-calculate Currents (Cheap)
    curr_q = np.einsum("bnd, hdk -> bnhk", x_in, params.W_Q)
    curr_k = np.einsum("bnd, hdk -> bnhk", x_in, params.W_K)
    curr_v = np.einsum("bnd, hdv -> bnhv", x_in, params.W_V)
    
    # 2. Dynamics with Error
    u_q, v_q, pv_q = gle_linear_dynamics(state.u_q, state.v_q, curr_q, err_q, hp)
    u_k, v_k, pv_k = gle_linear_dynamics(state.u_k, state.v_k, curr_k, err_k, hp)
    u_v, v_v, pv_v = gle_linear_dynamics(state.u_v, state.v_v, curr_v, err_v, hp)
    
    state = state._replace(
        u_q=u_q, v_q=v_q, prosp_v_q=pv_q,
        u_k=u_k, v_k=v_k, prosp_v_k=pv_k,
        u_v=u_v, v_v=v_v, prosp_v_v=pv_v
    )
    
    # 6. Gradient Accumulation (Hebbian)
    
    # dW_Y += pv_y * o.T
    dW_Y = np.einsum('bnhv, bnd -> hvd', o, pv_y)
    
    # dW_Q += pv_q * x_in.T
    # pv_q: (B, N, H, Dk)
    # x_in: (B, N, Dm)
    # dW_Q target: (H, Dm, Dk)
    # einsum: bnhk, bnd -> hdk
    dW_Q = np.einsum('bnhk, bnd -> hdk', pv_q, x_in)
    
    # dW_K, dW_V (Zero for now as error is 0, but mechanism exists)
    dW_K = np.einsum('bnhk, bnd -> hdk', pv_k, x_in)
    dW_V = np.einsum('bnhv, bnd -> hdv', pv_v, x_in)
    
    grads = GLEGradients(dW_Q, dW_K, dW_V, dW_Y)
    
    return state, r_y, grads

def init_gle_state(batch_size, hp: GLEHyperparameters, key):
    # Initialize zero arrays for all states
    N = hp.n_regions
    H = hp.n_heads
    Dk = hp.d_k
    Dv = hp.d_v
    Dm = hp.d_model
    
    def z(*dims): return np.zeros((batch_size, N, *dims))
    
    state = GLEState(
        u_q=z(H, Dk), u_k=z(H, Dk), u_v=z(H, Dv), u_y=z(Dm),
        v_q=z(H, Dk), v_k=z(H, Dk), v_v=z(H, Dv), v_y=z(Dm),
        prosp_v_q=z(H, Dk), prosp_v_k=z(H, Dk), prosp_v_v=z(H, Dv), prosp_v_y=z(Dm),
        M=np.zeros((batch_size, N, H, Dv, Dk)),
        history=None, step=0
    )
    return state
