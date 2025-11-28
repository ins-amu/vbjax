import jax
import jax.numpy as np
import optax
from vbjax.ct_mhsa import init_ct_mhsa, Hyperparameters, scan_ct_mhsa, NetworkState, CTMHSAParams

def make_loss_fn(hp: Hyperparameters):
    def loss_fn(params: CTMHSAParams, state: NetworkState, inputs: jax.Array, targets: jax.Array):
        # inputs: (T, B, N, D)
        # targets: (T, B, N, D)
        
        # Run model
        (final_state, final_y), (outputs, surprise) = scan_ct_mhsa(params, state, inputs, hp)
        
        # Predictive Coding Loss: 0.5 * || targets - outputs ||^2
        # targets are x_{t+1}, outputs are y_t.
        diff = targets - outputs
        loss = 0.5 * np.mean(diff**2)
        return loss
    return loss_fn

def run_training_check():
    print("Initializing Training Check...")
    # Hyperparameters
    hp = Hyperparameters(n_regions=5, n_heads=2, d_k=4, d_v=4, d_model=4, lam=0.9)
    batch_size = 2
    seq_len = 20
    
    # Init
    key = jax.random.PRNGKey(100)
    k_p, k_d = jax.random.split(key)
    params, state, _ = init_ct_mhsa(hp, k_p, batch_size=batch_size)
    
    # Fake Data: Random sequence
    # We want to predict next step.
    data = jax.random.normal(k_d, (seq_len + 1, batch_size, hp.n_regions, hp.d_model))
    inputs = data[:-1] # 0..T-1
    targets = data[1:] # 1..T
    
    # Optimizer
    learning_rate = 1e-3
    # "Implement a learning rate warmup schedule"
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=10,
        decay_steps=100,
        end_value=1e-5
    )
    # "Gradient clipping (global norm)"
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule)
    )
    
    opt_state = optimizer.init(params)
    
    # Training Step
    loss_fn = make_loss_fn(hp)
    
    @jax.jit
    def train_step(params, state, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, state, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Starting training loop (Overfitting check)...")
    # Train on single batch to overfit
    for i in range(100):
        params, opt_state, loss = train_step(params, state, opt_state, inputs, targets)
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss:.6f}")
            
    print(f"Final Loss: {loss:.6f}")
    
    # Stability Check: Check M statistics
    print("Checking Memory Stability...")
    # Run for longer sequence
    long_inputs = jax.random.normal(jax.random.PRNGKey(1), (100, batch_size, hp.n_regions, hp.d_model))
    (final_state, _), _ = scan_ct_mhsa(params, state, long_inputs, hp)
    
    m_mean = np.mean(final_state.M)
    m_std = np.std(final_state.M)
    print(f"Memory Mean: {m_mean:.4f}, Std: {m_std:.4f}")
    
    if np.isnan(m_mean) or m_std > 100:
        print("WARNING: Memory Diverged!")
    else:
        print("Memory Stable.")

if __name__ == "__main__":
    run_training_check()
