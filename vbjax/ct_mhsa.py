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
    # Memory update-rate factor u = dt/tau (paper Alg. 1: M += (dt/tau)(v*phi(k)^T - M)).
    # Paper retention is lambda = 1 - dt/tau = 1 - lam; with leakage g_ell the recurrence is
    #   M_t = (1 - g_ell*lam) M_{t-1} + lam * v_t phi(k_t)^T   (paper eq. 1, g_ell=1)
    # Default lam=0.05 <=> retention lambda=0.95 (paper Shakespeare/translation settings: dt=50ms, tau~1s).
    lam: float = 0.05
    leak: float = 1.0  # leakage factor g_ell (paper Alg. 1/eq. 1); 1 = full leak
    tau: float = 0.0  # integration time constant in units of dt; 0 => update rate falls back to lam
    act: str = "relu"  # phi: "relu" (threshold-linear, paper default) or "elup1"
    use_block_residual: bool = False  # paper core has no in-block residual (residual is token-level)
    use_ln: bool = False  # paper core has no layer-norm inside the block
    sigma_a: float = 0.0
    delta_rule: int = 0   # 1 => error-gated memory write M += u(v - M phi(k)) phi(k)^T  # apical-afferent width; 0 => identity (no A), >0 => Gaussian A
    dt: float = 0.1
    v_c: float = 10.0
    steps_per_token: int = 1

    @property
    def memory_update_rate(self) -> float:
        """Update-rate factor u = dt/tau (what the recurrence actually multiplies the delta by).

        If tau > 0, u = dt/tau (paper convention). If tau == 0 (default), the legacy `lam`
        field is used directly as the update rate u (backward compatible with the pre-paper core).
        Retention lambda (paper) = 1 - g_ell * u.
        """
        if self.tau > 0:
            return self.dt / self.tau
        return self.lam

    @property
    def retention(self) -> float:
        """Paper retention lambda = 1 - g_ell * u."""
        return 1.0 - self.leak * self.memory_update_rate


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
    A: jax.Array  # (H, d_k, d_k) apical afferent matrix (Gaussian on diagonal); identity if off
    ln_gamma: Optional[jax.Array] = None  # (d_model,)
    ln_beta: Optional[jax.Array] = None  # (d_model,)


def make_apical_matrix(sigma: float, d_k: int, n_heads: int = 1) -> jax.Array:
    """Gaussian apical-afferent matrix A per head (paper eq. after eq. 5, fig. 2d).

    A_ij = exp(-(i-j)^2 / (2 sigma^2)). With sigma=0 this degenerates to identity
    (the plain linear-attention / 'transformer id' case). Returns (H, d_k, d_k).
    """
    if sigma <= 0:
        return np.broadcast_to(np.eye(d_k), (n_heads, d_k, d_k))
    idx = np.arange(d_k)
    diff = idx[:, None] - idx[None, :]
    A = np.exp(-(diff**2) / (2.0 * sigma**2))
    return np.broadcast_to(A, (n_heads, d_k, d_k))


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

    A = make_apical_matrix(hp.sigma_a, hp.d_k, hp.n_heads)

    params = CTMHSAParams(
        W_Q=W_Q, W_K=W_K, W_V=W_V, W_Y=W_Y, C=C, A=A, ln_gamma=ln_gamma, ln_beta=ln_beta
    )

    # Initialize State M
    # Shape: (B, N, H, Dv, Dk)
    M_shape = (batch_size, hp.n_regions, hp.n_heads, hp.d_v, hp.d_k)
    M = np.zeros(M_shape)

    state = NetworkState(M=M, history=history, step=0)

    return params, state, lags


# Phase 2: The Cortical Microcircuit (The "Head")


def compute_projections(
    params: CTMHSAParams, x: jax.Array, act: str = "relu"
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

    # Activation phi on keys and queries (paper: monotone increasing, e.g. threshold-linear).
    # Default "relu" is the threshold-linear choice; "elup1" is a smooth, strictly-positive variant
    # often used for stability in linear attention.
    if act == "elup1":
        k = jax.nn.elu(k) + 1.0
        q = jax.nn.elu(q) + 1.0
    else:  # relu (threshold-linear)
        k = jax.nn.relu(k)
        q = jax.nn.relu(q)

    return q, k, v


def update_memory_l23(state: NetworkState, k: jax.Array, v: jax.Array, hp: Hyperparameters) -> Tuple[NetworkState, jax.Array]:
    """
    Update memory M using the paper's recurrent key-value update (eq. 1, Alg. 1).
      M_t = M_{t-1} + (dt/tau) * ( v_t phi(k_t)^T - g_ell * M_{t-1} )
    Returns state and surprise (Frobenius norm of the delta).
    """
    # k: (B, N, H, Dk)
    # v: (B, N, H, Dv)
    # M: (B, N, H, Dv, Dk)

    u = hp.memory_update_rate  # dt/tau
    g = hp.leak  # g_ell

    if getattr(hp, "delta_rule", 0):
        # Delta-rule / error-gated write: store only the retrieval RESIDUAL
        # (what memory fails to predict about the new key-value pair):
        #   M += u * (v - M phi(k)) phi(k)^T
        # This is the local prediction-error plasticity form; the plain
        # outer-product accumulation is recovered when M=0 each step.
        o_pred = np.einsum('bnhvk,bnhk->bnhv', state.M, k)   # memory's prediction of v
        target = np.einsum('bnhv,bnhk->bnhvk', v - o_pred, k)
    else:
        # Outer product v * phi(k)^T -> (B, N, H, Dv, Dk)
        target = np.einsum('bnhv,bnhk->bnhvk', v, k)

    # Delta M = (dt/tau) * (target - g_ell * M_prev)
    delta_M = u * (target - g * state.M)
    M_new = state.M + delta_M

    # Surprise: Frobenius norm over last two dims (Dv, Dk); shape (B, N, H)
    surprise = np.linalg.norm(delta_M, axis=(-2, -1))

    return state._replace(M=M_new), surprise


def retrieve_query_l5(params: CTMHSAParams, state: NetworkState, q: jax.Array) -> jax.Array:
    """
    Retrieve output from memory:  o = M * (A * phi(q))
    (paper eq. 5 with the apical afferent matrix A; A=identity recovers the plain case).
    """
    # M: (B, N, H, Dv, Dk)
    # q: (B, N, H, Dk) (already phi-activated)
    # A: (H, Dk, Dk) with indices (h, i, k)
    # Out: (B, N, H, Dv)
    Aq = np.einsum('hik,bnhk->bnhi', params.A, q)  # (B, N, H, Dk); per-head A applied to q
    o = np.einsum('bnhvk,bnhk->bnhv', state.M, Aq)
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
    Single time step of MHSA block (paper eqs. 1-3).
    Returns: state, y, surprise
    """
    q, k, v = compute_projections(params, x, act=hp.act)

    # Update Memory (L2/3)
    state_new, surprise = update_memory_l23(state, k, v, hp)

    # Retrieve (L5) via apical afferent matrix A
    o = retrieve_query_l5(params, state_new, q)

    # Aggregate
    y = aggregate_heads(o, params)

    # Optional in-block residual + layer norm (OFF by default: paper core has neither;
    # residual is token-level x_t + y_t and normalisation happens at the embedding/decoding stage).
    if hp.use_block_residual:
        y = y + x
    if hp.use_ln and params.ln_gamma is not None and params.ln_beta is not None:
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
