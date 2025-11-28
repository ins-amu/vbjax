import jax
import jax.numpy as np
from vbjax.ct_mhsa import init_ct_mhsa, Hyperparameters, mhsa_step, NetworkState, CTMHSAParams
import numpy as onp

def test_init_and_structure():
    hp = Hyperparameters(n_regions=10, n_heads=4, d_k=8, d_v=8, d_model=8, lam=0.5)
    key = jax.random.PRNGKey(0)
    params, state = init_ct_mhsa(hp, key, batch_size=2)
    
    # Check structure
    print("Params structure:", jax.tree_util.tree_structure(params))
    print("State structure:", jax.tree_util.tree_structure(state))
    
    # Check shapes
    assert params.W_Q.shape == (4, 8, 8)
    assert state.M.shape == (2, 10, 4, 8, 8)
    
    # Check memory size roughly
    # 2 * 10 * 4 * 8 * 8 * 4 bytes = ~20KB. Tiny. 
    # For full size: 1 * 84 * 8 * 16 * 16 * 4 = 688KB. Perfectly fine.

def test_needle_in_haystack():
    """
    Feed a distinct (k, v) pair at t=0.
    Feed empty inputs for t=1..10.
    Query with q approx k at t=11.
    Assert output matches v (decayed).
    """
    # Setup single head, single region for simplicity or just standard dims
    hp = Hyperparameters(n_regions=1, n_heads=1, d_k=4, d_v=4, d_model=4, lam=0.5)
    key = jax.random.PRNGKey(1)
    params, state = init_ct_mhsa(hp, key, batch_size=1)
    
    # Manually set weights to identity to make reasoning easier
    # W_Q, W_K, W_V, W_Y = Identity-like
    # Actually, let's just call update_memory and retrieve directly to test logic 
    # without weight interference, or set weights to Identity.
    
    # Let's test the core functions directly first? 
    # No, let's use mhsa_step but override params to Identity to isolate logic.
    
    # Construct Identity params
    # W (H, Dm, Dk). Dm=4, Dk=4.
    I = np.eye(4).reshape(1, 4, 4)
    params = params._replace(W_Q=I, W_K=I, W_V=I, W_Y=I)
    
    # Distinct K, V
    # x_0 = [1, 0, 0, 0] -> k=[1,0,0,0], v=[1,0,0,0]
    x0 = np.array([[[1.0, 0.0, 0.0, 0.0]]]) # (B=1, N=1, D=4)
    
    # Step 0
    state, y0 = mhsa_step(params, state, x0, hp)
    
    # Check M state
    # M_0 = (1-lam)*0 + lam * (v*k^T)
    # lam=0.5. M_0 = 0.5 * [[1,0,0,0]^T * [1,0,0,0]] = 0.5 * TopLeft 1.
    assert state.M[0,0,0,0,0] == 0.5
    
    # Steps 1..10: Input zero
    x_zero = np.zeros_like(x0)
    for _ in range(10):
        state, _ = mhsa_step(params, state, x_zero, hp)
        
    # M should decay by (1-lam)=0.5 each time.
    # After 10 steps: 0.5 * (0.5)^10
    expected_val = 0.5 * (0.5)**10
    assert np.allclose(state.M[0,0,0,0,0], expected_val)
    
    # Step 11: Query
    # q = [1, 0, 0, 0]. So x=[1,0,0,0].
    # But we want to retrieve, not write new stuff effectively?
    # mhsa_step does both. 
    # If we input x=[1,0,0,0], we add another memory trace AND retrieve.
    # The retrieval uses the Updated memory.
    # So we retrieve (M_decayed + new_trace) * q.
    # We want to see the old trace.
    # The new trace will be 0.5 * (1*1) = 0.5.
    # The old trace is small.
    # This test setup in the plan says "Feed empty inputs... Query with q".
    # If we feed q via x, we also generate k and v.
    # If we want to ONLY query, we need to suppress k/v or have separate API?
    # The model `mhsa_step` is coupled.
    # "Query with q approx k".
    # If we set W_V to zero for this step? Or x such that k=0 but q!=0?
    # Impossible if W_Q=W_K=I and input is same x.
    # We need x such that x*W_Q = q, x*W_K = 0.
    # If W_Q != W_K, possible.
    # Let's modify W_K for the last step or use orthogonal inputs?
    # Let's use orthogonal input for query?
    # No, we want to retrieve the OLD key.
    # So we send q = old_k.
    # If we send x corresponding to old_k, we write old_v again.
    # That dominates the signal.
    
    # To strictly test "Needle in Haystack" on this *coupled* model:
    # We rely on the fact that we want to retrieve what was stored.
    # If we write it again, we retrieve (old + new).
    # If we want to verify "memory retention", we check if result > new_only.
    
    state_before = state
    state, y_out = mhsa_step(params, state, x0, hp)
    
    # Expected output y:
    # M_new = 0.5*M_old + 0.5*(v*k^T)
    # y = M_new * q
    #   = 0.5*M_old*q + 0.5*(v*k^T)*q
    #   = Signal + Noise/New
    # We can check if Signal component is present.
    
    # Alternatively, since we have access to internal functions, we can test `retrieve_query_l5` directly
    # without updating memory, to match the spirit of the test "Query ...".
    # The plan says "Validation Step 2: Check ... logic ... Query with q".
    # I will use the internal functions for this specific test to verify the memory mechanism.
    
    from vbjax.ct_mhsa import retrieve_query_l5, compute_projections
    
    q, k, v = compute_projections(params, x0)
    o = retrieve_query_l5(state_before, q)
    
    # o should be M_before * q.
    # M_before is approx 0.5^11 * (v0 * k0^T).
    # o = 0.5^11 * v0 * (k0^T * q).
    # Since q = k0, k0^T*q = |k0|^2 = 1.
    # So o approx 0.5^11 * v0.
    
    print("Output magnitude:", np.linalg.norm(o))
    print("Expected magnitude:", 0.5 * (0.5)**10)
    assert np.allclose(o, v * (0.5 * (0.5)**10))
    print("Needle in Haystack passed.")

def test_gradients():
    hp = Hyperparameters()
    key = jax.random.PRNGKey(2)
    params, state = init_ct_mhsa(hp, key)
    x = jax.random.normal(key, (1, hp.n_regions, hp.d_model))
    
    def loss_fn(params, state, x):
        state_new, y = mhsa_step(params, state, x, hp)
        return np.sum(y**2)
    
    grads = jax.grad(loss_fn)(params, state, x)
    # Check if grads are not zero
    assert np.any(grads.W_Q != 0)
    print("Gradients computed successfully.")

if __name__ == "__main__":
    test_init_and_structure()
    test_needle_in_haystack()
    test_gradients()
