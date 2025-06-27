from functools import partial
import numpy as np
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.98'

import tqdm
import jax
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import jax_dataclasses as jdc
import jax.nn as jnn
from jax.example_libraries import optimizers


# Set random seed for reproducibility
key = random.PRNGKey(0)

# --- 1. Define Data Structures ---
@jdc.pytree_dataclass
class ModelParams:
    # Input Embeddings
    patch_embed_w: jnp.ndarray   # NEW: (patch_dim_in, embed_dim)
    patch_embed_b: jnp.ndarray   # NEW: (embed_dim,)
    pos_embeds: jnp.ndarray      # (16, 16, embed_dim)
    seq_embeds: jnp.ndarray      # (seq_len, embed_dim)
    
    # DeltaNet Layers
    delta_layers: list           # List of layer-specific parameters
    
    # Output Head
    output_w: jnp.ndarray        # (embed_dim, 1)
    output_b: jnp.ndarray

@jdc.pytree_dataclass
class DeltaLayerParams:
    wq: jnp.ndarray              # (num_heads, embed_dim, head_dim)
    wk: jnp.ndarray              # (num_heads, embed_dim, head_dim)
    wv: jnp.ndarray              # (num_heads, embed_dim, head_dim)
    wo: jnp.ndarray              # (num_heads * head_dim, embed_dim)
    norm: jnp.ndarray
    B: jnp.ndarray               # (num_heads,) - NEW: Learnable per-head parameter
    alpha: jnp.ndarray           # (num_heads,) - NEW: Learnable per-head precision amplification

@jdc.pytree_dataclass
class TrainingConfig:
    # Configuration values are fixed during training and not differentiated.
    # We mark them as `Static` so JAX treats them as part of the Pytree's
    # structure (treedef) rather than as dynamic leaf nodes. This is crucial
    # for JIT compilation efficiency.
    learning_rate: jdc.Static[float]
    batch_size: jdc.Static[int]
    num_iterations: jdc.Static[int]

    # Model dimensions are static for a given training run.
    embed_dim: jdc.Static[int]
    num_heads: jdc.Static[int]
    num_layers: jdc.Static[int]
    patch_size: jdc.Static[int]
    image_size: jdc.Static[int]
    # Precision amplification factor is a static hyperparameter.
    # This will now serve as the *initialization* value for the learnable alphas.
    alpha: jdc.Static[float]

# --- 2. Model Initialization ---

def init_model(key, config, scale=1e-6):
    head_dim = config.embed_dim // config.num_heads
    seq_len = (config.image_size // config.patch_size)**2
    patch_dim_in = config.patch_size * config.patch_size # 4*4 = 16 in your config
    
    # Adjusted key split to account for new patch_embed_w and patch_embed_b
    keys = random.split(key, 4 + config.num_layers * 7)

    randn = lambda key, shape: random.normal(key, shape)*scale
    
    delta_layers = []
    for i in range(config.num_layers):
        layer_params = DeltaLayerParams(
            wq=randn(keys[i*7+0], (config.num_heads, config.embed_dim, head_dim)),
            wk=randn(keys[i*7+1], (config.num_heads, config.embed_dim, head_dim)),
            wv=randn(keys[i*7+2], (config.num_heads, config.embed_dim, head_dim)),
            wo=randn(keys[i*7+3], (config.num_heads * head_dim, config.embed_dim)),
            norm=jnp.ones(config.embed_dim),
            B=jnp.ones(config.num_heads)*0.1, # Existing: Initialize B for each head
            alpha=jnp.full((config.num_heads,), config.alpha) # NEW: Initialize per-head alpha
        )
        delta_layers.append(layer_params)
        
    return ModelParams(
        patch_embed_w=randn(keys[-6], (patch_dim_in, config.embed_dim)), # NEW
        patch_embed_b=randn(keys[-5], (config.embed_dim,)),             # NEW
        pos_embeds=randn(keys[-4], (config.image_size // config.patch_size, config.image_size // config.patch_size, config.embed_dim)),
        seq_embeds=randn(keys[-3], (seq_len, config.embed_dim)),
        delta_layers=delta_layers,
        output_w=randn(keys[-2], (config.embed_dim, 1)),
        output_b=randn(keys[-1], (1,))
    )

# --- 3. Core Model Logic ---

def z_score(x, axis=-1, eps=1e-5):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def delta_net_layer_forward(params: DeltaLayerParams, x: jnp.ndarray, S: jnp.ndarray):
    
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
        def head_update(S_h, B_h, alpha_h, q_token, k_token, v_token):
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
            temp = jnp.exp(-alpha_h * jnp.linalg.norm(error)) + 1e-6
            
            # Efferent signal
            efferent = jax.nn.softmax((S_curr @ q_token) / temp)
            return S_curr, efferent

        # Passed params.alpha to vmap
        S_next, efferent_h = vmap(head_update)(S_prev, params.B, params.alpha, q_h, k_h, v_h)
        
        # Concatenate heads and project out
        efferent_flat = efferent_h.flatten()
        output_token = efferent_flat @ params.wo
        
        # Skip connection
        final_token = token_x + output_token
        
        return S_next, final_token

    ok = True  # for debugging
    if ok:
        final_S, output_sequence = jax.lax.scan(scan_fn, S, x) #, unroll=16)
    else:
        S_curr = S
        output_tokens = []
        for i, token_x in enumerate(x):
            S_next, final_token = scan_fn(S_curr, token_x)
            S_curr = S_next
            output_tokens.append(final_token)
            jax.debug.print("\ttoken {i} {Sn} {fn}", i=i, Sn=jnp.linalg.norm(S_curr), fn=jnp.linalg.norm(final_token))
        final_S = S_curr
        output_sequence = jnp.stack(output_tokens)

    return output_sequence, final_S

def forward_pass(params: ModelParams, batch_patches: jnp.ndarray, config: TrainingConfig):
    # Project raw patches to embed_dim
    # batch_patches shape: (batch_size, seq_len, patch_size*patch_size)
    # patch_embed_w shape: (patch_size*patch_size, embed_dim)
    # Resulting patch_embeddings shape: (batch_size, seq_len, embed_dim)
    patch_embeddings = jnp.einsum('bsf,fd->bsd', batch_patches, params.patch_embed_w) + params.patch_embed_b

    # Add positional and sequential embeddings
    batch_size, seq_len, _ = patch_embeddings.shape # Use shape of patch_embeddings for consistency
    pos_embeds_flat = params.pos_embeds.reshape(seq_len, config.embed_dim)
    x = patch_embeddings + pos_embeds_flat + params.seq_embeds

    # Initialize recurrent states (one per layer)
    head_dim = config.embed_dim // config.num_heads
    initial_S = jnp.zeros((config.num_layers, config.num_heads, head_dim, head_dim))
    
    # Propagate through layers
    current_x = x
    for i in range(config.num_layers):
        # print(i, seq_len)
        layer_params = params.delta_layers[i]
        S_layer = initial_S[i]
        
        # Vmap across the batch dimension
        batch_layer_fwd = vmap(delta_net_layer_forward, in_axes=(None, 0, None))
        current_x, _ = batch_layer_fwd(layer_params, current_x, S_layer)

    # Readout from the last token
    final_representation = current_x[:, -1] # (batch, embed_dim)
    logits = final_representation @ params.output_w + params.output_b
    return jnp.squeeze(logits)

# --- 4. Loss and Training Functions ---
def loss_fn(params: ModelParams, batch_patches, batch_labels, config: TrainingConfig):
    logits = forward_pass(params, batch_patches, config)

    # Manually implement sigmoid binary cross-entropy for numerical stability
    # log(sigmoid(x)) = log_sigmoid(x)
    # log(1 - sigmoid(x)) = log_sigmoid(-x)
    loss = - (batch_labels * jnn.log_sigmoid(logits) +
              (1 - batch_labels) * jnn.log_sigmoid(-logits))

    loss = jnp.mean(loss) # Take the mean over the batch
    return loss

@partial(jit, static_argnums=(0, 4, 5, 6)) # step, config, opt_update_fn, get_params_fn are static
def train_step(step: int, params: ModelParams, opt_state, batch, config: TrainingConfig, opt_update_fn, get_params_fn):
    batch_patches, batch_labels = batch
    loss_val, grads = jax.value_and_grad(loss_fn, has_aux=False)(params, batch_patches, batch_labels, config)
    opt_state = opt_update_fn(step, grads, opt_state)
    params = get_params_fn(opt_state)
    return params, opt_state, loss_val

def train(config: TrainingConfig, model_params: ModelParams, train_images, train_labels):
    print("--- Starting Training ---")
    opt_init, opt_update, get_params = optimizers.adam(config.learning_rate)
    opt_state = opt_init(model_params)
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

    for i in (pbar := tqdm.trange(config.num_iterations, ncols=60)):
        # print(i, 'create minibatch')
        key, subkey = random.split(key)
        indices = random.permutation(subkey, train_images.shape[0])[:config.batch_size]
        batch_images = train_images[indices]
        batch_labels = train_labels[indices]
        
        # print(i, 'preprocess')
        batch_patches = preprocess_images(batch_images)
        
        # print(i, 'train step')
        model_params, opt_state, loss_val = train_step(i, model_params, opt_state, (batch_patches, batch_labels), config, opt_update, get_params)
        
        msg = f"Loss: {loss_val:.4f}"
        pbar.set_description(msg)
            
    print("--- Training Finished ---")
    return model_params


# --- 5. Main Execution ---
if __name__ == '__main__':

    # Configuration
    config = TrainingConfig(
        learning_rate=5e-4,
        batch_size=256,
        num_iterations=10001,
        embed_dim=512,
        num_heads=16,
        num_layers=6,
        patch_size=8,
        image_size=64,
        alpha=1.0,
    )

    # Initialize model
    key = random.PRNGKey(0)
    model_params = init_model(key, config)

    # Load data
    images_data = jnp.array(np.load('ims.npy'))
    labels_data = jnp.array(np.load('masks.npy'))
    print(f"Loaded data: images shape {images_data.shape}, labels shape {labels_data.shape}")

    # Run training
    trained_params = train(config, model_params, images_data, labels_data)

# sits at 0.693 which is -log(50%) for binary cross-entropy loss, indicating a random guess or chance-level performance.