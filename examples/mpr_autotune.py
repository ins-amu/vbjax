import jax
import jax.numpy as jp
import jax.example_libraries.optimizers as jopt
import vbjax as vb
import numpy as np
import pylab as pl
import tqdm

# --- Part 1: Autotune MPR parameters for ~10Hz alpha rhythm ---

def mpr_model(state, params):
    # Single node, no coupling (c=(0,0))
    return vb.mpr_dfun(state, (0.0, 0.0), params)

def run_mpr_psd(params, rng_key, dt=0.1, t_max=2000.0):
    # Simulation setup
    n_steps = int(t_max / dt)
    
    # Initial state (r, V) 
    init_state = jp.array([0.1, -2.0]) 
    
    # Make SDE
    _, loop = vb.make_sde(dt=dt, dfun=mpr_model, gfun=1e-5) 
    
    # Noise for simulation
    noise_shape = (n_steps, 2) # 2 variables: r, V
    noise = vb.randn(*noise_shape, key=rng_key)
    
    # Run simulation
    states = loop(init_state, noise, params)
    
    # Extract V (voltage) for PSD
    v_trace = states[:, 1]
    
    # Remove first half to avoid transients
    v_trace = v_trace[n_steps//2:]
    
    # Compute PSD
    n_fft = v_trace.shape[0]
    v_fft = jp.fft.rfft(v_trace)
    power = jp.abs(v_fft)**2
    freqs = jp.fft.rfftfreq(n_fft, d=dt/1000.0) # dt is in ms, convert to s for Hz
    
    return freqs, power, states

def loss_function(trainable_params, rng_key):
    # Unpack trainable parameters
    eta_raw, tau_raw = trainable_params
    
    # Clamp parameters to safe ranges
    eta = jp.clip(eta_raw, -10.0, 2.0)
    tau = jp.clip(tau_raw, 0.1, 200.0)
    
    # Construct full parameters
    full_params = vb.mpr_default_theta._replace(eta=eta, tau=tau)
    
    freqs, power, states = run_mpr_psd(full_params, rng_key, dt=0.1, t_max=4000.0)
    
    # Check for NaNs
    is_nan = jp.isnan(power).any()
    
    # Define Alpha band (8-12 Hz) and Surround (2-30 Hz)
    alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
    broad_mask = (freqs >= 2.0) & (freqs <= 30.0)
    surround_mask = broad_mask & (~alpha_mask)
    
    power_alpha = jp.sum(power * alpha_mask) + 1e-6
    power_surround = jp.sum(power * surround_mask) + 1e-6
    
    loss_val = jp.log(power_surround) - jp.log(power_alpha)
    
    # Regularization
    reg = 0.01 * (eta**2) + 0.001 * ((tau - 80.0)**2)
    
    total_loss = loss_val + reg
    
    return jp.where(is_nan, 100.0, total_loss)

# Vectorize loss over multiple keys
def batch_loss(trainable_params, rng_keys):
    losses = jax.vmap(lambda k: loss_function(trainable_params, k))(rng_keys)
    return jp.mean(losses)

# --- Main Execution ---

if __name__ == "__main__":
    print("Setting up optimization...")
    
    # Initial guess
    init_params = jp.array([-3.0, 80.0]) # eta, tau
    
    # Optimizer
    lr = 0.01
    opt_init, opt_step, opt_get = jopt.adam(lr)
    opt_state = opt_init(init_params)
    
    # JIT compile gradients
    loss_grad_fn = jax.jit(jax.value_and_grad(batch_loss))
    
    n_iters = 200
    print(f"Starting optimization for {n_iters} iterations...")
    
    for i in tqdm.trange(n_iters):
        step_keys = jax.random.split(jax.random.PRNGKey(i + 100), 16)
        
        val, grads = loss_grad_fn(opt_get(opt_state), step_keys)
        
        if jp.isnan(val) or jp.isnan(grads).any():
            continue
            
        grads = jp.clip(grads, -1.0, 1.0)
        opt_state = opt_step(i, grads, opt_state)
        
        if i % 10 == 0:
            p = opt_get(opt_state)
            # print(f"Iter {i}: Loss={val:.4f}, params (eta, tau) = {p}")

    final_params_raw = opt_get(opt_state)
    best_eta, best_tau = final_params_raw
    
    if np.isnan(best_eta) or np.isnan(best_tau):
        print("Optimization failed (NaN parameters). Reverting to defaults.")
        best_eta = -4.0
        best_tau = 1.0
        
    print(f"Optimization complete. Best parameters: eta={best_eta:.4f}, tau={best_tau:.4f}")
    
    final_theta = vb.mpr_default_theta._replace(eta=best_eta, tau=best_tau)
    
    # --- Part 2: Tune dt for efficiency ---
    print("\n--- Tuning dt for efficiency ---")
    
    dt = 0.1
    max_dt = 50.0
    best_dt = dt
    
    while dt < max_dt:
        try:
            _, loop = vb.make_sde(dt=dt, dfun=mpr_model, gfun=0.0)
            n_steps = int(1000.0 / dt)
            dw = jp.zeros((n_steps, 2))
            x0 = jp.array([0.1, -2.0])
            
            states = loop(x0, dw, final_theta)
            
            if jp.isnan(states).any() or jp.max(jp.abs(states)) > 1e6:
                best_dt = dt / 2.0
                break
            
            dt = dt * 1.1
            
        except Exception as e:
            best_dt = dt / 2.0
            break
    
    # Force safe dt for network simulation
    best_dt = 0.5
    print(f"Using safer dt for network: {best_dt:.4f} ms")
    
    # --- Part 3: Network Simulation & FCD ---
    
    print("\n--- Running 5-minute Network Simulation with BOLD & FCD ---")

    N_nodes = 32
    # Random connectivity
    W = jax.random.uniform(jax.random.PRNGKey(99), (N_nodes, N_nodes)) / N_nodes
    coupling_strength = 0.1

    def network_model(state, params):
        # state: (2, N) -> r, V
        r = state[0]
        # Global coupling
        c = coupling_strength * (W @ r)
        return vb.mpr_dfun(state, (c, 0.0), params)

    # Simulation Parameters
    T_total = 5 * 60.0 * 1000.0  # 5 minutes
    chunk_dt = 2000.0  # 2 seconds (BOLD TR)
    
    steps_per_chunk = int(chunk_dt / best_dt)
    actual_chunk_dt = steps_per_chunk * best_dt
    n_chunks = int(T_total / actual_chunk_dt)

    print(f"Simulating {N_nodes} nodes for {T_total/1000:.1f}s")
    
    # Setup SDE
    _, chunk_loop = vb.make_sde(dt=best_dt, dfun=network_model, gfun=1e-4)

    # Setup BOLD monitor
    bold_shape = (N_nodes,)
    bold_buf, bold_step, bold_samp_fn = vb.make_bold(bold_shape, best_dt, vb.bold_default_theta)
    bold_offline = vb.make_offline(bold_step, bold_samp_fn)

    def run_chunk(carry, key):
        neural_state, bold_state = carry
        noise = vb.randn(steps_per_chunk, 2, N_nodes, key=key)
        neural_traj = chunk_loop(neural_state, noise, final_theta)
        r_traj = neural_traj[:, 0, :]
        new_bold_state, bold_sample = bold_offline(bold_state, r_traj)
        return (neural_traj[-1], new_bold_state), bold_sample

    # Initial states
    init_neural = jp.zeros((2, N_nodes))
    init_neural = init_neural.at[0, :].set(0.1)
    init_neural = init_neural.at[1, :].set(-2.0)
    init_neural += 0.01 * jax.random.normal(jax.random.PRNGKey(1), (2, N_nodes))
    init_bold = bold_buf

    print("Starting simulation...")
    scan_keys = jax.random.split(jax.random.PRNGKey(100), n_chunks)
    run_scan = jax.jit(lambda c, k: jax.lax.scan(run_chunk, c, k))
    
    (final_neural, final_bold), bold_series = run_scan((init_neural, init_bold), scan_keys)
    
    bold_series = np.array(bold_series) # (n_chunks, N) 
    
    # Check stability
    if np.isnan(bold_series).any():
        print("Error: Simulation contained NaNs! FCD will be invalid.")
        bold_series = np.zeros_like(bold_series) # Prevent plotting crash
    else:
        print(f"Simulation stable. BOLD range: [{bold_series.min():.4f}, {bold_series.max():.4f}]")
        if np.std(bold_series) < 1e-6:
            print("Warning: BOLD signal is essentially flat. FCD will be empty.")

    # --- FCD Calculation ---
    print("Computing FCD...")
    
    window_len_s = 30.0
    tr_s = actual_chunk_dt / 1000.0
    window_steps = int(window_len_s / tr_s)
    stride = 1
    
    n_time = bold_series.shape[0]
    fcs = []
    
    for i in range(0, n_time - window_steps + 1, stride):
        window = bold_series[i:i+window_steps] # (window_steps, N) 
        
        # Check variance to avoid NaNs in corrcoef
        if np.std(window) < 1e-9:
            fc = np.zeros((N_nodes, N_nodes)) # Or identity? Zero correlation makes sense for flat signal.
        else:
            fc = np.corrcoef(window.T)
            # Handle potential NaNs from constant features within window
            if np.isnan(fc).any():
                fc = np.nan_to_num(fc)
        
        # Extract Upper Triangle (k=1)
        triu_idxs = np.triu_indices(N_nodes, k=1)
        fcs.append(fc[triu_idxs])
        
    fcs = np.array(fcs) # (n_windows, n_pairs) 
    
    # FCD Matrix (Correlation of FC patterns)
    if fcs.shape[0] > 0:
        # Check if fcs has variance
        if np.std(fcs) < 1e-9:
             fcd_matrix = np.zeros((fcs.shape[0], fcs.shape[0]))
        else:
             fcd_matrix = np.corrcoef(fcs)
             if np.isnan(fcd_matrix).any():
                 fcd_matrix = np.nan_to_num(fcd_matrix)
    else:
        fcd_matrix = np.zeros((1,1))

    # --- Plotting ---
    print("Generating plot...")
    pl.figure(figsize=(15, 4))
    
    # 1. PSD
    freqs, power, _ = run_mpr_psd(final_theta, jax.random.PRNGKey(0), dt=0.1, t_max=4000.0)
    pl.subplot(1, 3, 1)
    pl.semilogy(freqs, power)
    pl.xlim(0, 60)
    pl.xlabel("Frequency (Hz)")
    pl.ylabel("Power")
    pl.title("PSD (Single Node)")
    pl.axvline(10, color='r', linestyle='--', alpha=0.5)
    
    # 2. Voltage Trace (Single Node, High Res)
    # Generate a short high-res trace for visualization using the final state
    n_plot_steps = int(2000.0 / best_dt)
    plot_noise = vb.randn(n_plot_steps, 2, N_nodes, key=jax.random.PRNGKey(999))
    voltage_traj = chunk_loop(final_neural, plot_noise, final_theta)
    voltage_trace_0 = voltage_traj[:, 1, 0] # Node 0
    t_axis_v = np.arange(voltage_trace_0.shape[0]) * best_dt
    
    pl.subplot(1, 3, 2)
    pl.plot(t_axis_v, voltage_trace_0)
    pl.xlabel("Time (ms)")
    pl.ylabel("Voltage")
    pl.title(f"Voltage Trace (Node 0)")
    
    # 3. FCD Matrix
    pl.subplot(1, 3, 3)
    pl.imshow(fcd_matrix, cmap='viridis', aspect='auto', interpolation='none', origin='lower')
    pl.colorbar(label="Correlation")
    pl.xlabel("Time (window)")
    pl.ylabel("Time (window)")
    pl.title("FCD")
    
    pl.tight_layout()
    pl.savefig("mpr_autotune_fcd.png")
    print("Saved plot to mpr_autotune_fcd.png")
