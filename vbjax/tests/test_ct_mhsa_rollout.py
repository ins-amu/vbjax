import jax
import jax.numpy as np
from vbjax.ct_mhsa import init_ct_mhsa, Hyperparameters, scan_ct_mhsa, NetworkState, CTMHSAParams, mhsa_step, network_coupling

def test_stability():
    T = 100
    hp = Hyperparameters(n_regions=2, n_heads=2, d_k=4, d_v=4, d_model=4, lam=0.1) # Fast decay
    key = jax.random.PRNGKey(42)
    params, state = init_ct_mhsa(hp, key, batch_size=1)
    
    # Make C Identity to avoid mixing for this check
    params = params._replace(C=np.eye(hp.n_regions))
    
    # Input pulse
    inputs = np.zeros((T, 1, hp.n_regions, hp.d_model))
    inputs = inputs.at[0, :, :, :].set(1.0)
    
    (final_state, final_y), outputs = scan_ct_mhsa(params, state, inputs, hp)
    
    # Check for NaNs
    assert not np.isnan(outputs).any()
    assert not np.isnan(final_state.M).any()
    
    # Check decay
    # Max output at T=0 or T=1 should be higher than at T=99
    max_early = np.max(np.abs(outputs[:5]))
    max_late = np.max(np.abs(outputs[-5:]))
    
    print(f"Max Early: {max_early}, Max Late: {max_late}")
    assert max_late < max_early
    print("Stability test passed.")

def test_python_loop_comparison():
    T = 5
    hp = Hyperparameters(n_regions=2, n_heads=2, d_k=4, d_v=4, d_model=4, lam=0.5)
    key = jax.random.PRNGKey(43)
    params, state = init_ct_mhsa(hp, key, batch_size=1)
    
    inputs = jax.random.normal(key, (T, 1, hp.n_regions, hp.d_model))
    
    # JAX Scan
    (scan_final_state, scan_final_y), scan_outputs = scan_ct_mhsa(params, state, inputs, hp)
    
    # Python Loop
    state_curr = state
    y_curr = np.zeros((1, hp.n_regions, hp.d_model))
    loop_outputs = []
    
    for t in range(T):
        x_t = network_coupling(y_curr, params, inputs[t])
        state_curr, y_curr = mhsa_step(params, state_curr, x_t, hp)
        loop_outputs.append(y_curr)
        
    loop_outputs = np.stack(loop_outputs)
    
    # Compare
    print("Max diff:", np.max(np.abs(scan_outputs - loop_outputs)))
    assert np.allclose(scan_outputs, loop_outputs, atol=1e-5)
    print("Python loop comparison passed.")

if __name__ == "__main__":
    test_stability()
    test_python_loop_comparison()
