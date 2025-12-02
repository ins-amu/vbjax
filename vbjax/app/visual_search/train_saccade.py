import jax
import jax.numpy as np
import optax
import argparse
import pickle
import os
import numpy as onp
from vbjax.app.visual_search.model import NetworkState, agent_step
from vbjax.app.visual_search.train import get_oracle_saccade, extract_patches
from vbjax.app.visual_search.assessment_probes import load_trained_params

def train_saccade_cloning():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3) # Higher LR for rapid adaptation
    args = parser.parse_args()
    
    # Load Params
    print("Loading params...")
    params, state_proto, hp = load_trained_params()
    if params is None:
        print("Failed to load params.")
        return

    # Data Generators
    from vbjax.app.visual_search.data import generate_dataset, make_scanpaths
    
    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr)
    )
    opt_state = optimizer.init(params)
    
    # Loss Function (Single Step Saccade Cloning)
    # We don't need full rollout for this. We can just sample random positions and train the instantaneous vector field.
    
    @jax.jit
    def train_step(params, opt_state, images, masks, pos, tasks):
        # 1. Extract Patches
        patches = extract_patches(images, pos, hp.patch_size)
        
        # 2. Oracle Saccades
        target_deltas = get_oracle_saccade(pos, masks, None, hp.patch_size, images.shape)
        
        # 3. Agent Step (We reuse the state but don't care about memory update for this)
        # We need a batch of states.
        B = images.shape[0]
        M_batch = np.repeat(state_proto.M, B, axis=0)
        lags = state_proto.lags
        # Dummy history
        curr_state = NetworkState(M=M_batch, history=None, step=0, lags=lags)
        if state_proto.history is not None:
             hist_batch = np.repeat(state_proto.history, B, axis=1)
             curr_state = curr_state._replace(history=hist_batch)
             
        # Run Agent
        # step_idx=0 (static time embedding)
        _, (logits, saccade_pred, _, _, _) = agent_step(
            params, curr_state, patches, pos, tasks, 0, hp
        )
        
        # Loss
        loss = np.mean((saccade_pred - target_deltas)**2)
        
        # Grads
        grads = jax.grad(lambda p: np.mean((agent_step(p, curr_state, patches, pos, tasks, 0, hp)[1][1] - target_deltas)**2))(params)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss

    print("Starting Saccade Cloning...")
    
    # Data Loop
    key = jax.random.PRNGKey(999)
    
    for i in range(args.steps):
        # Generate on the fly
        images_np, tasks_np, _, masks_np = generate_dataset(n_samples=args.batch_size)
        
        # Random positions in [-1, 1]
        key, k_pos = jax.random.split(key)
        pos = jax.random.uniform(k_pos, (args.batch_size, 2), minval=-0.8, maxval=0.8)
        
        images = jax.device_put(images_np) / 255.0
        masks = jax.device_put(masks_np)
        tasks = jax.device_put(tasks_np.flatten())
        
        params, opt_state, loss = train_step(params, opt_state, images, masks, pos, tasks)
        
        if i % 100 == 0:
            print(f"Step {i:04d}: Saccade MSE = {loss:.6f}")
            
    print("Saving refined parameters...")
    with open("visual_search_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Done.")

if __name__ == "__main__":
    train_saccade_cloning()
