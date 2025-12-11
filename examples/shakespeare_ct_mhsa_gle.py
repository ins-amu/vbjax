import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from tqdm import tqdm
from vbjax.ct_mhsa_gle import GLEHyperparameters, CTMHSAParams, GLEState, init_gle_state, gle_mhsa_step, phi

# --- Data ---
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
    starts = jax.random.randint(key, (batch_size,), 0, len(data) - seq_len - 1)
    x_batch = []
    y_batch = []
    for start in starts:
        start = int(start)
        x = data[start : start + seq_len]
        y = data[start + 1 : start + seq_len + 1]
        x_batch.append(x)
        y_batch.append(y)
    return np.stack(x_batch), np.stack(y_batch)

# --- Initialization ---
def init_params(key, hp: GLEHyperparameters, vocab_size):
    k_q, k_k, k_v, k_y, k_c, k_emb, k_head = jax.random.split(key, 7)
    
    def init_w(k, shape, scale=1.0):
        lim = np.sqrt(6 / sum(shape[-2:]))
        return jax.random.uniform(k, shape, minval=-lim, maxval=lim) * scale

    W_Q = init_w(k_q, (hp.n_heads, hp.d_model, hp.d_k))
    W_K = init_w(k_k, (hp.n_heads, hp.d_model, hp.d_k))
    W_V = init_w(k_v, (hp.n_heads, hp.d_model, hp.d_v))
    W_Y = init_w(k_y, (hp.n_heads, hp.d_v, hp.d_model))
    C = jax.random.normal(k_c, (hp.n_regions, hp.n_regions)) * 0.1
    
    mhsa_params = CTMHSAParams(W_Q, W_K, W_V, W_Y, C)
    
    # Embedding & Head (Static for now or Hebbian?)
    # Let's keep Embedding/Head standard dense for simplicity, 
    # but we need to learn them. 
    # To be "Full GLE", we should treat them as layers.
    # For this prototype, let's cheat slightly and use SGD on Embed/Head 
    # driven by the final error, while the Core learns via GLE.
    # actually, let's treat Head as just a linear projection W_Head.
    
    embed = jax.random.normal(k_emb, (vocab_size, hp.d_model)) * 0.02
    head = jax.random.normal(k_head, (hp.d_model, vocab_size)) * 0.02
    
    return mhsa_params, embed, head

# --- Training Loop ---

def train_shakespeare_gle():
    print("Loading data...")
    try:
        data, vocab_size, idx_to_char = load_data("examples/input.txt")
    except FileNotFoundError:
        print("Creating dummy input.txt")
        with open("examples/input.txt", "w") as f:
            f.write("To be or not to be, that is the question. " * 100)
        data, vocab_size, idx_to_char = load_data("examples/input.txt")

    print(f"Data size: {len(data)}, Vocab: {vocab_size}")
    
    # Hyperparams
    hp = GLEHyperparameters(
        d_model=32, d_k=16, d_v=16, n_heads=4, n_regions=4,
        dt=0.2, tau_m=5.0, gamma=0.5, lam=0.9, lr_w=0.001
    )
    
    batch_size = 16
    seq_len = 32
    n_steps = 500
    
    key = jax.random.PRNGKey(0)
    params, embed, head = init_params(key, hp, vocab_size)
    
    # Optimizers for Embed/Head (Standard SGD)
    lr_global = 0.01
    
    print("Starting GLE Training...")
    
    losses = []
    
    for i in tqdm(range(n_steps)):
        key, k_batch = jax.random.split(key)
        x_batch, y_batch = get_batch(data, batch_size, seq_len, k_batch)
        
        # Init State
        state = init_gle_state(batch_size, hp, key)
        
        # Accumulators
        total_dW_Y = np.zeros_like(params.W_Y)
        total_dW_Q = np.zeros_like(params.W_Q)
        total_dW_K = np.zeros_like(params.W_K)
        total_dW_V = np.zeros_like(params.W_V)
        
        total_dHead = np.zeros_like(head)
        total_dEmbed = np.zeros_like(embed)
        
        batch_loss = 0
        
        # Sequence Loop
        for t in range(seq_len):
            xt = x_batch[:, t] # (B,)
            yt = y_batch[:, t] # (B,)
            
            # 1. Embedding
            x_emb = embed[xt] # (B, D)
            
            # Input Injection (Region 0)
            # x_in: (B, N, D)
            x_in = np.zeros((batch_size, hp.n_regions, hp.d_model))
            x_in = x_in.at[:, 0, :].set(x_emb)
            
            # 2. Compute Error at Output (Region 3 - Last Region)
            # Current Output Activity of Core
            # We use state.u_y (converted to r) from PREVIOUS step or CURRENT?
            # GLE usually runs dynamics then updates.
            # Let's run the step first with 0 error to get activity, 
            # then calc error? No, error drives dynamics.
            # We use the previous step's prediction to compare against current target.
            
            r_out = phi(state.u_y[:, hp.n_regions - 1, :]) # (B, D) at Region N-1
            
            # Predict Logic (Head)
            logits = r_out @ head # (B, Vocab)
            
            # True Target One Hot
            y_onehot = jax.nn.one_hot(yt, vocab_size)
            
            # Error Vector (Target - Prediction)
            # This is the "Dopamine/Error" signal
            # Standard CrossEntropy Grad ~ (p - y)
            probs = jax.nn.softmax(logits)
            err_logits = (y_onehot - probs) # "Target - Pred" direction
            
            # Backprop Error to Region N-1 (Head -> Core)
            # e_core = err_logits @ head.T
            e_core_out = err_logits @ head.T # (B, D)
            
            # Inject this error into GLE step
            error_in = np.zeros((batch_size, hp.n_regions, hp.d_model))
            error_in = error_in.at[:, hp.n_regions - 1, :].set(e_core_out)
            
            # 3. GLE Step
            state, r_y, grads = gle_mhsa_step(params, state, x_in, error_in, hp)
            
            # 4. Accumulate Gradients
            
            # Core
            total_dW_Y += grads.dW_Y
            total_dW_Q += grads.dW_Q
            total_dW_K += grads.dW_K
            total_dW_V += grads.dW_V
            
            # Head (Standard Hebbian: Error * Input)
            # dHead = r_out.T @ err_logits
            dHead = r_out.T @ err_logits
            total_dHead += dHead
            
            # Embedding (Standard Backprop-ish: Error * Input_Index)
            # Error at input region 0?
            # We need to look at prosp_v at region 0
            # e_emb = state.prosp_v_y[:, 0, :] # (B, D)
            # For now, let's skip embedding training to keep it simple or use simple Hebbian
            
            # Loss Calc (Cross Entropy)
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=-1))
            batch_loss += loss
            
        # Update Weights (End of Sequence)
        # Normalize
        norm = 1.0 / (batch_size * seq_len)
        
        params = params._replace(
            W_Y = params.W_Y + hp.lr_w * total_dW_Y * norm,
            W_Q = params.W_Q + hp.lr_w * total_dW_Q * norm,
            W_K = params.W_K + hp.lr_w * total_dW_K * norm,
            W_V = params.W_V + hp.lr_w * total_dW_V * norm
        )
        head = head + lr_global * total_dHead * norm
        
        avg_loss = batch_loss / seq_len
        losses.append(avg_loss)
        
        if i % 50 == 0:
            grad_norm = np.linalg.norm(total_dW_Y * norm)
            print(f"Step {i}: Loss {avg_loss:.4f} | |dW_Y|: {grad_norm:.6f}")
            
    # Plot
    plt.plot(losses)
    plt.title("GLE Shakespeare Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("gle_shakespeare_loss.png")
    print("Saved plot to gle_shakespeare_loss.png")

if __name__ == "__main__":
    train_shakespeare_gle()
