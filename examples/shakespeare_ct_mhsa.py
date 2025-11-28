from typing import NamedTuple
import jax
import jax.numpy as np
import optax
import numpy as onp
import os
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

def init_shakespeare(key, hp: Hyperparameters, vocab_size):
    k_e, k_m, k_h = jax.random.split(key, 3)
    
    # Embedding
    embed = jax.random.normal(k_e, (vocab_size, hp.d_model)) * 0.02
    
    # MHSA
    mhsa_params, state = init_ct_mhsa(hp, k_m, batch_size=1) # Batch size for state doesn't matter here as we vmap or scan over batch differently? 
    # Wait, scan_ct_mhsa handles batch in inputs (T, B, N, D).
    # State M needs to be initialized with (B, N, H, Dv, Dk).
    # We'll handle state init in the forward pass or reset it.
    
    # Head
    head = jax.random.normal(k_h, (hp.d_model, vocab_size)) * 0.02
    
    return ShakespeareParams(embed, mhsa_params, head)

def shakespeare_forward(params: ShakespeareParams, state: NetworkState, x_indices: jax.Array, hp: Hyperparameters):
    # x_indices: (B, T)
    # Embed: (B, T, D)
    # Transpose to (T, B, D) for scan_ct_mhsa?
    # Actually scan_ct_mhsa expects inputs (T, B, N, D).
    # Here N=1.
    
    B, T = x_indices.shape
    
    # Embedding lookup
    x_emb = params.embed[x_indices] # (B, T, D)
    x_emb = np.transpose(x_emb, (1, 0, 2)) # (T, B, D)
    x_emb = np.expand_dims(x_emb, 2) # (T, B, 1, D)
    
    # MHSA
    (final_state, _), mhsa_out = scan_ct_mhsa(params.mhsa, state, x_emb, hp)
    # mhsa_out: (T, B, 1, D)
    
    # Head
    # Flatten: (T*B, D)
    mhsa_out_flat = mhsa_out.reshape(-1, hp.d_model)
    logits = mhsa_out_flat @ params.head # (T*B, Vocab)
    vocab_size = params.head.shape[1]
    logits = logits.reshape(T, B, vocab_size)
    logits = np.transpose(logits, (1, 0, 2)) # (B, T, Vocab)
    
    return logits, final_state

def loss_fn_shakespeare(params, state, x, y, hp):
    logits, _ = shakespeare_forward(params, state, x, hp)
    # Cross Entropy
    # y: (B, T)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    loss = -np.mean(np.sum(one_hot * log_probs, axis=-1))
    return loss

def train_shakespeare():
    print("Loading data...")
    data, vocab_size, idx_to_char = load_data("examples/input.txt")
    print(f"Data size: {len(data)}, Vocab: {vocab_size}")
    
    # Hyperparameters
    # Small model for speed check
    hp = Hyperparameters(
        n_regions=1, # Single region for sequence
        n_heads=4,
        d_k=16,
        d_v=16,
        d_model=32,
        lam=0.95
    )
    batch_size = 32
    seq_len = 64
    learning_rate = 1e-3
    
    key = jax.random.PRNGKey(42)
    k_init, k_train = jax.random.split(key)
    
    params = init_shakespeare(k_init, hp, vocab_size)
    
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)
    
    # Init state (B, N, H, Dv, Dk)
    # We need to re-init state for each batch or carry it?
    # Usually for truncated BPTT we carry it, but for random batches we zero it.
    # Let's zero it for simplicity.
    def get_zero_state(batch_size):
        M = np.zeros((batch_size, hp.n_regions, hp.n_heads, hp.d_v, hp.d_k))
        return NetworkState(M)

    @jax.jit
    def train_step(params, opt_state, x, y):
        state = get_zero_state(x.shape[0])
        loss, grads = jax.value_and_grad(loss_fn_shakespeare)(params, state, x, y, hp)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Starting training...")
    for i in range(501): # Short run
        key, k_batch = jax.random.split(key)
        x_batch, y_batch = get_batch(data, batch_size, seq_len, k_batch)
        
        params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
        
        if i % 50 == 0:
            print(f"Iter {i}, Loss: {loss:.4f}")

    # Generation
    print("Generating text...")
    start_char = onp.array([[data[0]]]) # (1, 1)
    chars = [idx_to_char[data[0]]]
    
    state = get_zero_state(1)
    curr_x = start_char
    
    # Generate 100 chars
    # We need a generation step function that is not scan, or use scan 1 by 1?
    # We can use mhsa_step directly.
    
    # Need to handle embedding and head manually for generation step
    # Or reuse forward with T=1.
    
    gen_params = params
    
    for _ in range(200):
        logits, state = shakespeare_forward(gen_params, state, curr_x, hp)
        # logits: (1, 1, Vocab)
        next_char_idx = np.argmax(logits[0, 0]) # Greedy
        chars.append(idx_to_char[int(next_char_idx)])
        curr_x = np.array([[next_char_idx]])
        
    print("Generated:", "".join(chars))

if __name__ == "__main__":
    train_shakespeare()
