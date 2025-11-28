import jax
import jax.numpy as np
from vbjax.ct_mhsa import Hyperparameters, init_ct_mhsa, scan_ct_mhsa

def test_delayed_propagation():
    # Setup
    dt = 1.0 # 1 ms step
    v_c = 1.0 # 1 m/ms velocity
    dist = 5.0 # 5 ms delay
    
    hp = Hyperparameters(
        n_regions=2,
        n_heads=1,
        d_model=1,
        d_k=1,
        d_v=1,
        dt=dt,
        v_c=v_c
    )
    
    lengths = np.array([[0.0, dist], [dist, 0.0]]) # Distance matrix
    # Delay 0->1 is 5. 1->0 is 5.
    
    key = jax.random.PRNGKey(42)
    
    # Initial C: C_ij is from j to i.
    # We want 0 -> 1. So C_10 = 1.0.
    C = np.array([[0.0, 0.0], [1.0, 0.0]])
    
    params, state = init_ct_mhsa(hp, key, batch_size=1, initial_c=C, lengths=lengths)
    
    # Force W parameters to be positive to ensure signal propagation through ReLU
    # Use larger weights to avoid attenuation
    ones_shape = lambda x: np.ones(x.shape) * 0.5
    params = params._replace(
        W_Q=ones_shape(params.W_Q),
        W_K=ones_shape(params.W_K),
        W_V=ones_shape(params.W_V),
        W_Y=ones_shape(params.W_Y),
        ln_gamma=None,
        ln_beta=None
    )

    # params.lags check
    print("Lags:", params.lags)
    assert params.lags[1, 0] == 5
    
    # Run simulation
    T = 20
    inputs = np.zeros((T, 1, 2, 1))
    # Inject pulse at t=0 into node 0
    inputs = inputs.at[0, 0, 0, 0].set(10.0)
    
    (final_state, _), outputs = scan_ct_mhsa(params, state, inputs, hp)
    # outputs: (T, B, N, D)
    
    y0 = outputs[:, 0, 0, 0]
    y1 = outputs[:, 0, 1, 0]
    
    print("Node 0 output:", y0)
    print("Node 1 output:", y1)
    
    # Check timing
    # Node 0 should react at t=0 (instant)
    # Note: y_0 = MHSA(x_0). x_0 = input_0 = 10.
    # y_0 should be non-zero (unless random weights kill it completely).
    assert np.abs(y0[0]) > 1e-5
    
    # Node 1 should NOT react before t=5 (steps 0,1,2,3,4)
    # x_t[1] = C_10 * y_{t-5}[0].
    # For t < 5, t-5 < 0. y_{-1..-5} are 0 (init).
    # So x_t[1] = 0. y_t[1] = MHSA(0) = 0 (if bias is 0, which it is).
    
    assert np.all(np.abs(y1[:5]) < 1e-5)
    
    # At t=5, x_5[1] = y_0[0].
    # y_5[1] = MHSA(y_0[0]).
    # If y_0[0] != 0, y_5[1] should be != 0.
    assert np.abs(y1[5]) > 1e-5

if __name__ == "__main__":
    test_delayed_propagation()
