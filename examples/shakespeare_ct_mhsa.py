from typing import NamedTuple, Optional
import jax
import jax.numpy as np
import optax
import numpy as onp
import os
import matplotlib.pyplot as plt
from vbjax.ct_mhsa import init_ct_mhsa, Hyperparameters, scan_ct_mhsa, NetworkState, CTMHSAParams

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
    
    params, lags = init_shakespeare(k_init, hp, vocab_size, lengths, c_mask)
    
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
    
    for i in range(total_steps + 1):
        seq_len = 8 if i < 200 else 64
        
        key, k_batch = jax.random.split(key)
        x_batch, y_batch = get_batch(data, batch_size, seq_len, k_batch)
        
        params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch, state_train)
        
        if i % 100 == 0:
            print(f"Iter {i}, SeqLen {seq_len}, Loss: {loss:.4f}")

    # Generation
    print("Generating text with Top-K Sampling...")
    start_char = onp.array([[data[0]]]) # (1, 1)
    chars = [idx_to_char[data[0]]]
    
    # Init gen state (B=1)
    _, gen_state, _ = init_ct_mhsa(hp, k_init, batch_size=1, lengths=lengths)
    curr_x = start_char
    
    gen_params = params
    
    # Top-K Config
    top_k = 15
    
    all_surprises = []
    
    for _ in range(200):
        logits, gen_state, surprise = shakespeare_forward(gen_params, gen_state, curr_x, hp, lags)
        # surprise: (1, K, 1, N, H)
        all_surprises.append(surprise[0, :, 0, :, :])
        
        # logits: (1, 1, Vocab)
        next_token_logits = logits[0, 0]
        
        key, k_gen = jax.random.split(key)
        
        # Top-K Masking
        top_k_vals, _ = jax.lax.top_k(next_token_logits, top_k)
        min_val = top_k_vals[-1]
        mask = next_token_logits >= min_val
        logits_masked = np.where(mask, next_token_logits, -np.inf)
        
        # Sample
        next_char_idx = jax.random.categorical(k_gen, logits_masked)
        
        chars.append(idx_to_char[int(next_char_idx)])
        curr_x = np.array([[next_char_idx]])
        
    print("Generated:", "".join(chars))
    
    # Plotting Surprise
    try:
        # Stack: (200, K, N, H)
        surprises_arr = np.stack(all_surprises) 
        # Flatten time: (T_gen * K, N, H)
        surprises_flat = surprises_arr.reshape(-1, hp.n_regions, hp.n_heads)
        # Compute mean surprise across heads and regions
        neural_act = np.mean(surprises_flat, axis=(1, 2)) 
        
        plt.figure(figsize=(10, 6))
        plt.plot(neural_act)
        plt.title("Neural Surprise (Mean over Regions/Heads)")
        plt.xlabel("Micro-Steps")
        plt.ylabel("Surprise (Norm of Delta M)")
        plt.savefig("surprise_plot.png")
        print("Surprise plot saved to surprise_plot.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    train_shakespeare()