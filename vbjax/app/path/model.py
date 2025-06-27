from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax_dataclasses as jdc

# Set random seed for reproducibility
key = random.PRNGKey(0)

# setup model structure using jax dataclass
@jdc.pytree_dataclass
class Model:
    """
    DeltaNet-like model for solving pathfinder task
    """



# --- 1. Define Data Structures ---

@dataclass
class ModelParams:
    # Input Embeddings
    pos_embeds: jnp.ndarray      # (16, 16, embed_dim)
    seq_embeds: jnp.ndarray      # (seq_len, embed_dim)
    
    # DeltaNet Layers
    delta_layers: list           # List of layer-specific parameters
    
    # Output Head
    output_w: jnp.ndarray        # (embed_dim, 1)
    output_b: jnp.ndarray

@dataclass
class DeltaLayerParams:
    wq: jnp.ndarray              # (num_heads, embed_dim, head_dim)
    wk: jnp.ndarray              # (num_heads, embed_dim, head_dim)
    wv: jnp.ndarray              # (num_heads, embed_dim, head_dim)
    wo: jnp.ndarray              # (num_heads * head_dim, embed_dim)
    norm: jnp.ndarray
    B: jnp.ndarray               # (num_heads,) - NEW: Learnable per-head parameter

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    num_iterations: int
    # Model dimensions
    embed_dim: int
    num_heads: int
    num_layers: int
    patch_size: int
    image_size: int
    # Precision amplification factor
    alpha: float

# --- 2. Model Initialization ---

def init_model(key, config):
    head_dim = config.embed_dim // config.num_heads
    seq_len = (config.image_size // config.patch_size)**2
    
    keys = random.split(key, 2 + config.num_layers * 6) # Adjusted for new B param
    
    delta_layers = []
    for i in range(config.num_layers):
        layer_params = DeltaLayerParams(
            wq=random.normal(keys[i*6+0], (config.num_heads, config.embed_dim, head_dim)),
            wk=random.normal(keys[i*6+1], (config.num_heads, config.embed_dim, head_dim)),
            wv=random.normal(keys[i*6+2], (config.num_heads, config.embed_dim, head_dim)),
            wo=random.normal(keys[i*6+3], (config.num_heads * head_dim, config.embed_dim)),
            norm=jnp.ones(config.embed_dim),
            B=jnp.ones(config.num_heads) # NEW: Initialize B for each head
        )
        delta_layers.append(layer_params)
        
    return ModelParams(
        pos_embeds=random.normal(keys[-4], (config.image_size // config.patch_size, config.image_size // config.patch_size, config.embed_dim)),
        seq_embeds=random.normal(keys[-3], (seq_len, config.embed_dim)),
        delta_layers=delta_layers,
        output_w=random.normal(keys[-2], (config.embed_dim, 1)),
        output_b=random.normal(keys[-1], (1,))
    )

# --- 3. Core Model Logic ---

def z_score(x, axis=-1, eps=1e-5):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)

def delta_net_layer_forward(params: DeltaLayerParams, x: jnp.ndarray, S: jnp.ndarray, alpha: float):
    
    def scan_fn(carry, token_x):
        S_prev = carry
        
        # Z-score input token
        token_x_norm = z_score(token_x)
        
        # Project to Q, K, V for each head
        q_h = jnp.einsum('d,hdm->hm', token_x_norm, params.wq)
        k_h = jnp.einsum('d,hdm->hm', token_x_norm, params.wk)
        v_h = jnp.einsum('d,hdm->hm', token_x_norm, params.wv)
        
        # Sigmoid nonlinearity
        q_h, k_h = jax.nn.sigmoid(q_h), jax.nn.sigmoid(k_h)
        
        # --- Per-head recurrent update (vmapped) ---
        def head_update(S_h, B_h, q_token, k_token, v_token):
            # NEW: Prediction includes learnable parameter B
            v_pred = B_h * (S_h @ k_token)
            error = v_token - v_pred
            
            # State update
            dS = jnp.outer(error, k_token)
            S_curr = S_h + dS
            
            # --- Mathematical Equivalence Comment ---
            # The update S_curr = S_prev + outer(v - B * (S_prev @ k), k)
            # expands to: S_curr = S_prev + v @ k.T - B * S_prev @ k @ k.T
            # which rearranges to: S_curr = S_prev @ (I - B * k @ k.T) + v @ k.T
            # This is the DeltaNet update rule.
            # -----------------------------------------
            
            # Precision amplification
            temp = jnp.exp(-alpha * jnp.linalg.norm(error)) + 1e-6
            
            # Efferent signal
            efferent = jax.nn.softmax((S_curr @ q_token) / temp)
            return S_curr, efferent

        S_next, efferent_h = vmap(head_update)(S_prev, params.B, q_h, k_h, v_h)
        
        # Concatenate heads and project out
        efferent_flat = efferent_h.flatten()
        output_token = efferent_flat @ params.wo
        
        # Skip connection
        final_token = token_x + output_token
        
        return S_next, final_token

    final_S, output_sequence = jax.lax.scan(scan_fn, S, x)
    return output_sequence, final_S

def forward_pass(params: ModelParams, batch_patches: jnp.ndarray, config: TrainingConfig):
    # Add positional and sequential embeddings
    batch_size, seq_len, _ = batch_patches.shape
    pos_embeds_flat = params.pos_embeds.reshape(seq_len, config.embed_dim)
    x = batch_patches + pos_embeds_flat[jnp.newaxis, :, :] + params.seq_embeds[jnp.newaxis, :, :]

    # Initialize recurrent states (one per layer)
    head_dim = config.embed_dim // config.num_heads
    initial_S = jnp.zeros((config.num_layers, config.num_heads, head_dim, head_dim))
    
    # Propagate through layers
    current_x = x
    for i in range(config.num_layers):
        layer_params = params.delta_layers[i]
        S_layer = initial_S[i]
        
        # Vmap across the batch dimension
        batch_layer_fwd = vmap(delta_net_layer_forward, in_axes=(None, 0, None, None))
        current_x, _ = batch_layer_fwd(layer_params, current_x, S_layer, config.alpha)
        current_x = jax.nn.layer_norm(current_x, use_scale=False) * layer_params.norm
        
    # Readout from the mean of the final sequence
    final_representation = jnp.mean(current_x, axis=1) # (batch, embed_dim)
    logits = final_representation @ params.output_w + params.output_b
    return jnp.squeeze(logits)

# --- 4. Loss and Training Functions ---

def loss_fn(params: ModelParams, batch_patches, batch_labels, config: TrainingConfig):
    logits = forward_pass(params, batch_patches, config)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, batch_labels))
    return loss

@partial(jit, static_argnums=(3,))
def train_step(params: ModelParams, opt_state, batch, config: TrainingConfig):
    batch_patches, batch_labels = batch
    loss_val, grads = grad(loss_fn, has_aux=False)(params, batch_patches, batch_labels, config)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

def train(config: TrainingConfig, model_params: ModelParams, train_images, train_labels):
    print("--- Starting Training ---")
    opt_state = optimizer.init(model_params)
    key = random.PRNGKey(42)

    # Data preprocessing function
    def preprocess_images(images):
        p = config.patch_size
        b, h, w = images.shape
        # (B, H/p, W/p, p, p)
        patches = images.reshape(b, h // p, p, w // p, p).transpose(0, 1, 3, 2, 4)
        # (B, H/p * W/p, p*p)
        patches = patches.reshape(b, (h // p) * (w // p), p * p)
        return patches / 255.0

    for i in range(config.num_iterations):
        # Create a mini-batch
        key, subkey = random.split(key)
        indices = random.permutation(subkey, train_images.shape[0])[:config.batch_size]
        batch_images = train_images[indices]
        batch_labels = train_labels[indices]
        
        batch_patches = preprocess_images(batch_images)
        
        model_params, opt_state, loss_val = train_step(model_params, opt_state, (batch_patches, batch_labels), config)
        
        if i % 10 == 0:
            print(f"Iteration {i:03d} | Loss: {loss_val:.4f}")
            
    print("--- Training Finished ---")
    return model_params


# --- 5. Main Execution ---
if __name__ == '__main__':
    # Create dummy data files for demonstration
    print("Creating dummy data files 'ims.npy' and 'labels.npy'...")
    dummy_ims = np.random.randint(0, 256, size=(100, 64, 64), dtype=np.uint8)
    dummy_labels = np.random.randint(0, 2, size=(100,), dtype=np.int32)
    np.save('ims.npy', dummy_ims)
    np.save('labels.npy', dummy_labels)
    print("Dummy data created.")

    # Configuration
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=8,
        num_iterations=101,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        patch_size=4,
        image_size=64,
        alpha=1.0,
    )

    # Initialize optimizer
    optimizer = optax.adam(config.learning_rate)

    # Initialize model
    key = random.PRNGKey(0)
    model_params = init_model(key, config)

    # Load data
    try:
        images_data = np.load('ims.npy')
        labels_data = np.load('labels.npy')
        print(f"Loaded data: images shape {images_data.shape}, labels shape {labels_data.shape}")
    except FileNotFoundError:
        print("Error: Could not find 'ims.npy' or 'labels.npy'.")
        exit()

    # Run training
    trained_params = train(config, model_params, jnp.array(images_data, dtype=jnp.float32), jnp.array(labels_data))
