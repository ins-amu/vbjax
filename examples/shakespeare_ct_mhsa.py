from typing import NamedTuple, Optional
import jax
import jax.numpy as np
import optax
import numpy as onp
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vbjax.ct_mhsa import init_ct_mhsa, Hyperparameters, scan_ct_mhsa, NetworkState, CTMHSAParams

def save_checkpoint(params, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(params, f)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename="checkpoint.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

# 1. Data Pipeline
def load_data(path):
    with open(path, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    data_indices = onp.array([char_to_idx[ch] for ch in text], dtype=onp.int32)
    return data_indices, vocab_size, idx_to_char

def get_batch(data, batch_size, seq_len, key):
    # Random starts
    starts = jax.random.randint(key, (batch_size,), 0, len(data) - seq_len - 1)
    
    x_batch = []
    y_batch = []
    
    for start in starts:
        start = int(start)
        x = data[start : start + seq_len]
        y = data[start + 1 : start + seq_len + 1]
        x_batch.append(x)
        y_batch.append(y)
        
    return np.stack(x_batch), np.stack(y_batch) # (B, T)

# 2. Model Wrapper
class ShakespeareParams(NamedTuple):
    embed: jax.Array # (Vocab, D_model)
    mhsa: CTMHSAParams
    head: jax.Array # (D_model, Vocab)

def init_shakespeare(key, hp: Hyperparameters, vocab_size, lengths, c_mask):
    k_e, k_m, k_h, k_c = jax.random.split(key, 4)
    
    # Embedding
    embed = jax.random.normal(k_e, (vocab_size, hp.d_model)) * 0.02
    
    # Initialize C with mask
    # Random C (small init)
    C = jax.random.normal(k_c, (hp.n_regions, hp.n_regions)) * 0.1
    # Apply mask (Only allow connections on defined edges)
    C = C * c_mask
    
    # Initialize MHSA
    # init_ct_mhsa returns (params, state, lags)
    mhsa_params, state, lags = init_ct_mhsa(hp, k_m, batch_size=1, initial_c=C, lengths=lengths)
    
    # Head
    head = jax.random.normal(k_h, (hp.d_model, vocab_size)) * 0.02
    
    return ShakespeareParams(embed, mhsa_params, head), lags

def shakespeare_forward(params: ShakespeareParams, state: NetworkState, x_indices: jax.Array, hp: Hyperparameters, lags: jax.Array):
    # x_indices: (B, T)
    B, T = x_indices.shape
    
    # Embedding lookup
    x_emb = params.embed[x_indices] # (B, T, D)
    x_emb = np.transpose(x_emb, (1, 0, 2)) # (T, B, D)
    
    # Diamond Topology Input Injection
    # Inject ONLY into Region 0 (Visual)
    # Input shape for scan: (T, B, N, D)
    x_input = np.zeros((T, B, hp.n_regions, hp.d_model))
    x_input = x_input.at[:, :, 0, :].set(x_emb)
    
    # MHSA Scan
    (final_state, _), (mhsa_out, surprise_trace) = scan_ct_mhsa(params.mhsa, state, x_input, hp, lags=lags)
    # mhsa_out: (T, B, N, D)
    
    # Readout from Region 7 (Output of 8-region hierarchy)
    out_frontal = mhsa_out[:, :, 7, :] # (T, B, D)
    
    # Head Projection
    out_flat = out_frontal.reshape(-1, hp.d_model)
    logits = out_flat @ params.head # (T*B, Vocab)
    vocab_size = params.head.shape[1]
    logits = logits.reshape(T, B, vocab_size)
    logits = np.transpose(logits, (1, 0, 2)) # (B, T, Vocab)
    
    return logits, final_state, surprise_trace

def loss_fn_shakespeare(params, state, x, y, hp, lags):
    logits, _, _ = shakespeare_forward(params, state, x, hp, lags)
    # Cross Entropy
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    loss = -np.mean(np.sum(one_hot * log_probs, axis=-1))
    return loss

def train_shakespeare():
    print("Loading data...")
    data, vocab_size, idx_to_char = load_data("examples/input.txt")
    print(f"Data size: {len(data)}, Vocab: {vocab_size}")
    
    # Topology: Deep Hierarchy (8 Regions)
    # L1: 0 (In)
    # L2: 1, 2
    # L3: 3, 4
    # L4: 5, 6
    # L5: 7 (Out)
    # Bidirectional connections between levels.
    
    adj = onp.zeros((8, 8))
    
    # Helper to add bi-directional edge
    def connect(i, j):
        adj[i, j] = 1
        adj[j, i] = 1
        
    # L1 <-> L2
    connect(0, 1); connect(0, 2)
    # L2 <-> L3
    connect(1, 3); connect(1, 4)
    connect(2, 3); connect(2, 4)
    # L3 <-> L4
    connect(3, 5); connect(3, 6)
    connect(4, 5); connect(4, 6)
    # L4 <-> L5
    connect(5, 7); connect(6, 7)

    # Distances (Lengths) = 1.0 where connected
    lengths = adj.astype(np.float32)
    c_mask = adj.astype(np.float32)
    
    # Hyperparameters
    hp = Hyperparameters(
        n_regions=8,   # Increased to 8
        n_heads=8,
        d_k=32,
        d_v=32,
        d_model=128,
        lam=0.9,
        dt=1.0,
        v_c=1.0,
        steps_per_token=5 # 5 micro-steps per character
    )
    
    batch_size = 16   # Reduced from 32 to avoid OOM
    
    # Scheduler
    total_steps = 2000 # Increased from 500
    warmup_steps = 100
    learning_rate = 1e-3
    min_lr = 1e-4
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=min_lr,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=min_lr
    )
    
    key = jax.random.PRNGKey(42)
    k_init, k_train = jax.random.split(key)
    
    checkpoint_path = "checkpoint.pkl"
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        params = load_checkpoint(checkpoint_path)
        # We still need 'lags' which comes from init
        _, lags = init_shakespeare(k_init, hp, vocab_size, lengths, c_mask)
        do_train = False
    else:
        print("No checkpoint found. Initializing fresh parameters...")
        params, lags = init_shakespeare(k_init, hp, vocab_size, lengths, c_mask)
        do_train = True
    
    optimizer = optax.adamw(learning_rate=schedule)
    opt_state = optimizer.init(params)
    
    # Init state helper
    # Returns (params, state, lags)
    _, proto_state, _ = init_ct_mhsa(hp, k_init, batch_size=batch_size, lengths=lengths)
    
    def get_fresh_state(bs):
        return proto_state

    @jax.jit
    def train_step(params, opt_state, x, y, state_init):
        loss, grads = jax.value_and_grad(loss_fn_shakespeare)(params, state_init, x, y, hp, lags)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Starting training with Diamond Topology (4 Regions, Delays)...")
    print(f"Config: Batch={batch_size}, d_model={hp.d_model}, Steps={total_steps}, MicroSteps={hp.steps_per_token}")
    print("Curriculum: Short context (8) -> Long context (64)")
    
    state_train = get_fresh_state(batch_size)
    
    if do_train:
        for i in range(total_steps + 1):
            seq_len = 8 if i < 200 else 64
            
            key, k_batch = jax.random.split(key)
            x_batch, y_batch = get_batch(data, batch_size, seq_len, k_batch)
            
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch, state_train)
            
            if i % 100 == 0:
                print(f"Iter {i}, SeqLen {seq_len}, Loss: {loss:.4f}")
        
        save_checkpoint(params, checkpoint_path)
    else:
        print("Skipping training loop.")

    # Generation
    print("Generating text with Top-K Sampling (Chunked JIT)...")
    start_char = onp.array([[data[0]]]) # (1, 1)
    chars = [idx_to_char[data[0]]]
    
    # Init gen state (B=1)
    _, gen_state, _ = init_ct_mhsa(hp, k_init, batch_size=1, lengths=lengths)
    curr_x = start_char
    
    gen_params = params
    
    # Define JIT-compiled chunk generator
    @jax.jit
    def generate_chunk(state, start_token, key):
        def loop_fn(carry, _):
            curr_state, curr_tok, curr_key = carry
            
            # Forward pass (lags is captured from closure)
            logits, next_state, surprise = shakespeare_forward(gen_params, curr_state, curr_tok, hp, lags)
            
            # Sampling
            next_token_logits = logits[0, 0]
            curr_key, k_gen = jax.random.split(curr_key)
            
            # Top-K Masking
            top_k = 15
            top_k_vals, _ = jax.lax.top_k(next_token_logits, top_k)
            min_val = top_k_vals[-1]
            mask = next_token_logits >= min_val
            logits_masked = np.where(mask, next_token_logits, -np.inf)
            
            next_char_idx = jax.random.categorical(k_gen, logits_masked)
            next_tok = next_char_idx.reshape(1, 1)
            
            return (next_state, next_tok, curr_key), (next_char_idx, surprise)

        chunk_size = 32
        final_carry, (tokens, surprises) = jax.lax.scan(loop_fn, (state, start_token, key), None, length=chunk_size)
        return final_carry, tokens, surprises

    all_surprises = []
    num_chunks = 2000 // 32 + 1
    
    print(f"Generating ~{num_chunks * 32} tokens in {num_chunks} chunks...")
    
    key_gen = jax.random.PRNGKey(999) # separate key for generation
    
    for i in range(num_chunks):
        (gen_state, curr_x, key_gen), tokens, chunk_surprises = generate_chunk(gen_state, curr_x, key_gen)
        
        # Collect tokens
        for t in tokens:
            chars.append(idx_to_char[int(t)])
            
        # Collect surprises
        # chunk_surprises: (Chunk, 1, K, 1, N, H)
        # We want (Chunk, K, N, H)
        chunk_s = chunk_surprises[:, 0, :, 0, :, :]
        all_surprises.append(chunk_s)
        
        if i % 10 == 0:
            print(f"Chunk {i}/{num_chunks} done.")
        
    print("Generated:", "".join(chars[:200]) + "...")
    
    # Criticality Analysis
    try:
        # Stack: (NumChunks, ChunkSize, K, N, H)
        surprises_arr = np.concatenate(all_surprises, axis=0) # (TotalTokens, K, N, H)
        # Flatten time
        surprises_flat = surprises_arr.reshape(-1, hp.n_regions, hp.n_heads)
        # Compute mean surprise (Global Activity G)
        G = np.mean(surprises_flat, axis=(1, 2))
        
        # Convert to numpy for analysis/plotting
        G = onp.array(G)
        
        # Threshold (95th percentile)
        threshold = onp.percentile(G, 95)
        print(f"Surprise Threshold (95th percentile): {threshold:.4f}")
        
        # Binarize
        active = G > threshold
        
        # Find avalanches (contiguous segments)
        active_pad = onp.concatenate(([False], active, [False]))
        diff = onp.diff(active_pad.astype(int))
        starts = onp.where(diff == 1)[0]
        ends = onp.where(diff == -1)[0]
        
        durations = ends - starts
        sizes = []
        for s, e in zip(starts, ends):
            # Size = Area above threshold
            auc = onp.sum(G[s:e] - threshold)
            sizes.append(auc)
        sizes = onp.array(sizes)
        
        print(f"Found {len(sizes)} avalanches.")
        
        # Plot Distributions
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Time Series
        axes[0].plot(G[:1000]) # Show first 1000 micro-steps
        axes[0].axhline(threshold, color='r', linestyle='--')
        axes[0].set_title("Global Surprise G(t) (First 1000 steps)")
        axes[0].set_xlabel("Micro-Steps")
        axes[0].set_ylabel("Surprise")
        
        # Size Distribution (Log-Log)
        if len(sizes) > 0:
            # Use log bins
            bins = onp.logspace(onp.log10(max(1e-6, min(sizes))), onp.log10(max(sizes)), 20)
            hist, edges = onp.histogram(sizes, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            # Filter zeros
            mask = hist > 0
            axes[1].loglog(centers[mask], hist[mask], 'o-', label='Data')
            
            # Fit Power Law: log(P) = -alpha * log(S) + c
            if onp.sum(mask) > 2:
                log_x = onp.log10(centers[mask])
                log_y = onp.log10(hist[mask])
                slope, intercept = onp.polyfit(log_x, log_y, 1)
                alpha = -slope
                axes[1].plot(centers[mask], 10**intercept * centers[mask]**slope, 'r--', label=f'Fit $\\alpha={alpha:.2f}$')
            
            axes[1].set_title("Avalanche Size P(S)")
            axes[1].set_xlabel("Size (Area > Threshold)")
            axes[1].set_ylabel("Probability Density")
            axes[1].legend()
            
            # Duration Distribution (Log-Log)
            bins_d = onp.logspace(onp.log10(min(durations)), onp.log10(max(durations)), 20)
            hist_d, edges_d = onp.histogram(durations, bins=bins_d, density=True)
            centers_d = (edges_d[:-1] + edges_d[1:]) / 2
            mask_d = hist_d > 0
            axes[2].loglog(centers_d[mask_d], hist_d[mask_d], 'o-', label='Data')
            
            # Fit Power Law: log(P) = -beta * log(D) + c
            if onp.sum(mask_d) > 2:
                log_xd = onp.log10(centers_d[mask_d])
                log_yd = onp.log10(hist_d[mask_d])
                slope_d, intercept_d = onp.polyfit(log_xd, log_yd, 1)
                beta = -slope_d
                axes[2].plot(centers_d[mask_d], 10**intercept_d * centers_d[mask_d]**slope_d, 'r--', label=f'Fit $\\beta={beta:.2f}$')

            axes[2].set_title("Avalanche Duration P(D)")
            axes[2].set_xlabel("Duration (Micro-Steps)")
            axes[2].set_ylabel("Probability Density")
            axes[2].legend()
            
        plt.tight_layout()
        plt.savefig("avalanche_analysis.png")
        print("Avalanche analysis plot saved to avalanche_analysis.png")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_shakespeare()