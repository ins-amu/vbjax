"""
Delay-coupled Neural Mass Model Simulation with BOLD generation.
Performs a parameter sweep over Jsa using JAX Sharding (SPMD) and Autotuning.
"""

import os
import sys
import time
import tqdm
import numpy as np
import scipy.sparse
import pylab as pl
import jax
import jax.numpy as jp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import vbjax as vb
import gast_model as gm
import fused_jax

# Ensure XLA flags are set for multi-core if not already
try:
    n_cores = len(jax.devices())
    print(f"JAX sees {n_cores} devices.")
except:
    print("JAX setup issue.")

def benchmark_config(Seid70, idelays, init_state, bold_params, nn, n_svar, horizon, chunk_len, dt, num_skip, 
                     n_devices, batch_size):
    """
    Benchmarks a specific (n_devices, batch_size) configuration using JAX Sharding.
    Returns: microsteps_per_second
    """
    
    # 1. Define Sharding
    devices = jax.devices()[:n_devices]
    mesh = Mesh(devices, ('batch',))
    
    # Define Sharding Specs
    # Replicated: P()
    S_repl = NamedSharding(mesh, PartitionSpec())
    # Sharded on last axis (batch): P(None, ..., 'batch')
    
    step_fn = fused_jax.make_run_chunk(
        Seid70, idelays, horizon=horizon, chunk_len=chunk_len,
        dt=dt, num_skip=num_skip, num_item=batch_size, num_svar=n_svar,
        bold_params=bold_params
    )
    
    # Construct Dummy Data (Global Shape)
    buffer = jp.zeros((nn, horizon, batch_size)) + init_state[0]
    x = jp.zeros((n_svar, nn, batch_size)) + init_state[0]
    t = 0 # Scalar, replicated
    bold_init, _, _ = vb.make_bold((nn, batch_size), dt * num_skip * 1e-3, bold_params)
    
    # Params
    Jsa_dummy = jp.ones((batch_size,)) * 10.0
    
    base_theta = gm.dopa_default_theta._replace(
        I=0, Ja=0, Jsg=0, Jdopa=0, Jg=0, Rd=0, sigma_V=0.1,
        we=1e-4, wi=1e-4, wd=1e-4
    )
    
    params = base_theta._replace(Jsa=Jsa_dummy)

    try:
        if batch_size % n_devices != 0:
            return 0.0 # Cannot shard evenly
            
        # Shard Data
        # buffer: (nn, horizon, batch) -> P(None, None, 'batch')
        buffer = jax.device_put(buffer, NamedSharding(mesh, PartitionSpec(None, None, 'batch')))
        x = jax.device_put(x, NamedSharding(mesh, PartitionSpec(None, None, 'batch')))
        t = jax.device_put(t, S_repl)
        
        # bold_init is tuple of arrays. Most are (nodes, batch).
        bold_sharded = jax.tree_util.tree_map(
            lambda arr: jax.device_put(arr, NamedSharding(mesh, PartitionSpec(None, 'batch'))) if arr.ndim==2 else jax.device_put(arr, S_repl),
            bold_init
        )
        
        state = (buffer, x, t, bold_sharded)
        
        # Params: Jsa is (batch,), others scalar.
        params_sharded = jax.tree_util.tree_map(
             lambda arr: jax.device_put(arr, NamedSharding(mesh, PartitionSpec('batch'))) if (hasattr(arr, 'shape') and arr.shape==(batch_size,)) else jax.device_put(arr, S_repl),
             params
        )
        
        key = jax.random.PRNGKey(0)
        
        # Run JIT-compiled function
        # We don't need pmap. JIT + Sharded Inputs = SPMD.
        jit_step = jax.jit(step_fn)
        
        # Warmup
        jit_step(state, key, params_sharded)
        
        # Benchmark
        t0 = time.time()
        n_steps = 5
        final_state = state
        for _ in range(n_steps):
            final_state, _ = jit_step(final_state, key, params_sharded)
        jax.block_until_ready(final_state[1])
        dur = time.time() - t0
        
        total_microsteps = n_steps * chunk_len * batch_size 
        
        microsteps_per_sec = total_microsteps / dur
        return microsteps_per_sec
        
    except Exception as e:
        # print(f"  Failed {n_devices}x{batch_size}: {e}")
        return 0.0

def autotune(Seid70, idelays, init_state, bold_params, nn, n_svar, horizon, chunk_len, dt, num_skip):
    print("Autotuning configuration (1 min max)...")
    
    devices = jax.devices()
    n_dev_candidates = []
    # 1, 2, 4, 8 ...
    d = 1
    while d < len(devices):
        n_dev_candidates.append(d)
        d *= 2
    n_dev_candidates.append(len(devices))
    # Unique and sorted
    n_dev_candidates = sorted(list(set(n_dev_candidates)))[::-1]
    print(n_dev_candidates)
    
    batch_candidates = [4, 8, 16] 
    
    results = []
    
    t_start_tune = time.time()
    
    for nd in n_dev_candidates:
        for bs in batch_candidates:
            if time.time() - t_start_tune > 240:
                break
                
            if bs < nd: continue # inefficient
            
            metric = benchmark_config(
                Seid70, idelays, init_state, bold_params, nn, n_svar, horizon, chunk_len, dt, num_skip,
                nd, bs
            )
            
            if metric > 0:
                # Log in "k iter/s" or "M iter/s"
                msg = f"{metric/1e3:.1f}k" if metric < 1e6 else f"{metric/1e6:.2f}M"
                print(f"  Config {nd} devs, {bs} batch: {msg} microsteps/s")
                results.append({
                    'n_devices': nd,
                    'batch_size': bs,
                    'throughput': metric
                })
        if time.time() - t_start_tune > 120:
            print("  Autotune time limit reached.")
            break

    if not results:
        print("Autotune failed, defaulting to 1 dev, 16 batch")
        return 1, 16

    # Selection Logic
    # 1. Sort by throughput descending
    results.sort(key=lambda x: x['throughput'], reverse=True)
    best_thp = results[0]['throughput']
    
    # 2. Filter within 5% of best
    candidates = [r for r in results if r['throughput'] >= 0.95 * best_thp]
    
    # 3. Sort by 'efficiency': prefer fewer devices, then smaller batch
    candidates.sort(key=lambda x: (x['n_devices'], x['batch_size']))
    
    winner = candidates[0]
    print(f"Selected Config: {winner['n_devices']} devices, {winner['batch_size']} batch ({winner['throughput']/1e6:.2f}M steps/s)")
    
    return winner['n_devices'], winner['batch_size']

def main():
    # --- 1. Load Structural Connectivity ---
    W70 = np.loadtxt(f'Counts.csv')
    nn, _ = W70.shape
    np.fill_diagonal(W70, 0)
    L70 = np.loadtxt(f'Lengths.csv')

    # Create random masks (placeholder)
    Mi = np.random.rand(nn, nn) < 0.01
    Md = np.random.rand(nn, nn) < 0.01

    Ce70 = np.log(W70+1)/np.log(W70+1).max()
    Ci70 = Ce70 * Mi
    Cd70 = Ce70 * Md
    
    Ceid70 = np.vstack([Ce70, Ci70, Cd70])
    Leid70 = np.vstack([L70, L70, L70])
    Seid70 = scipy.sparse.csr_matrix(Ceid70)

    # --- 2. Compute Delays ---
    v_c = 10.0
    dt = 0.1
    idelays = (Leid70[Ceid70 != 0.0] / v_c / dt).astype(np.uint32)
    max_delay = idelays.max()

    # --- 3. Simulation Parameters ---
    n_svar = 7
    init_state = jp.array([0.01, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(n_svar, 1)
    
    # Goal: Complete Sweep
    TOTAL_ITEMS = 256
    Jsa_sweep = np.linspace(10, 50, TOTAL_ITEMS)
    sigma_V = 0.1

    bold_params = vb.bold_default_theta
    TR = 0.5 
    chunk_len = int(TR * 1000 / dt) 
    num_skip = 10 
    horizon = 512 

    # --- 4. Autotune ---
    best_nd, best_bs = autotune(
        Seid70, idelays, init_state, bold_params, nn, n_svar, horizon, chunk_len, dt, num_skip
    )
    
    # --- 5. Full Run ---
    print(f"Running full sweep ({TOTAL_ITEMS} items)...")
    
    # Re-setup Kernel with selected global batch size
    step_fn = fused_jax.make_run_chunk(
        Seid70, idelays, horizon=horizon, chunk_len=chunk_len,
        dt=dt, num_skip=num_skip, num_item=best_bs, num_svar=n_svar,
        bold_params=bold_params
    )
    jit_step = jax.jit(step_fn)
    
    # Sharding Mesh
    devices = jax.devices()[:best_nd]
    mesh = Mesh(devices, ('batch',))
    S_batch = NamedSharding(mesh, PartitionSpec('batch'))
    S_repl = NamedSharding(mesh, PartitionSpec())
    
    # Prepare Data Padding
    n_batches = int(np.ceil(TOTAL_ITEMS / best_bs))
    total_padded = n_batches * best_bs
    Jsa_padded = np.pad(Jsa_sweep, (0, total_padded - TOTAL_ITEMS), mode='edge')
    Jsa_batches = Jsa_padded.reshape(n_batches, best_bs)
    
    base_theta = gm.dopa_default_theta._replace(
        I=0, Ja=0, Jsg=0, Jdopa=0, Jg=0, Rd=0, sigma_V=0.1,
        we=1e-4, wi=1e-4, wd=1e-4
    )
    
    all_results = []
    
    for b in range(n_batches):
        # Prepare Batch Params
        batch_Jsa = Jsa_batches[b]
        
        # Shard Params
        # Jsa -> Shard across devices
        # Others -> Replicate
        params_tree = base_theta._replace(Jsa=batch_Jsa)
        params_sharded = jax.tree_util.tree_map(
             lambda arr: jax.device_put(arr, S_batch) if (hasattr(arr, 'shape') and arr.shape==(best_bs,)) else jax.device_put(arr, S_repl),
             params_tree
        )
        
        # Prepare State
        buffer = jp.zeros((nn, horizon, best_bs)) + init_state[0]
        x = jp.zeros((n_svar, nn, best_bs)) + init_state[0]
        t = 0
        bold_init, _, _ = vb.make_bold((nn, best_bs), dt * num_skip * 1e-3, bold_params)
        
        # Shard Data
        # buffer: (nn, horizon, batch) -> P(None, None, 'batch')
        buffer = jax.device_put(buffer, NamedSharding(mesh, PartitionSpec(None, None, 'batch')))
        x = jax.device_put(x, NamedSharding(mesh, PartitionSpec(None, None, 'batch')))
        t_sharded = jax.device_put(t, S_repl)
        
        bold_sharded = jax.tree_util.tree_map(
            lambda arr: jax.device_put(arr, NamedSharding(mesh, PartitionSpec(None, 'batch'))) if arr.ndim==2 else jax.device_put(arr, S_repl),
            bold_init
        )
        
        state = (buffer, x, t_sharded, bold_sharded)
        rng_key = jax.random.PRNGKey(42+b)
        
        # Run Simulation
        n_TR = 120
        batch_ts = []
        for i in tqdm.tqdm(range(n_TR), desc=f"Batch {b+1}/{n_batches}", leave=False):
             state, bold_val = jit_step(state, rng_key, params_sharded)
             batch_ts.append(bold_val)
             
        # Collect (move to host implicitly via numpy conversion later)
        bold_batch = jp.stack(batch_ts) # (Time, Nodes, Batch)
        all_results.append(bold_batch)

    # Concatenate
    full_bold = jp.concatenate(all_results, axis=2)
    full_bold = full_bold[:, :, :TOTAL_ITEMS]
    
    print(f"Sweep Complete. Shape: {full_bold.shape}")
    
    pl.figure(figsize=(10, 6))
    pl.title(f'BOLD Sweep (Jsa 10-50)')
    pl.imshow(full_bold.mean(axis=1), aspect='auto', extent=[10, 50, n_TR*TR, 0])
    pl.xlabel('Jsa')
    pl.ylabel('Time (s)')
    pl.colorbar(label='Mean BOLD')
    pl.savefig('bold_sweep_parallel.png')
    print("Saved bold_sweep_parallel.png")

if __name__ == "__main__":
    main()
