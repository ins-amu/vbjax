import jax
import jax.numpy as np
import numpy as onp
import pickle
import os
from vbjax.app.visual_search.model import init_visual_search, VisualSearchHyperparameters, agent_step, NetworkState, retina_forward, IDX_R_V1, IDX_R_PFC, IDX_R_FEF
from vbjax.ct_mhsa import Hyperparameters, mhsa_step
from vbjax.app.visual_search.train import extract_patches, make_rollout

def load_trained_params(path="visual_search_params.pkl"):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, None, None
        
    with open(path, "rb") as f:
        params = pickle.load(f)
        
    # Reconstruct HP (assuming defaults used in train.py)
    # Ideally we should have saved HP too, but we can reconstruct.
    n_r = 38
    mhsa_hp = Hyperparameters(
        n_regions=n_r, 
        n_heads=8,   
        d_k=16, d_v=16, d_model=64, 
        steps_per_token=5 # Match training default
    )
    hp = VisualSearchHyperparameters(
        mhsa=mhsa_hp,
        patch_size=64,
        n_tasks=2,
        n_classes=3,
        retina_channels=(16, 32) 
    )
    
    # Re-init state to get structure (values don't matter as we reset)
    key = jax.random.PRNGKey(0)
    # We need dummy weights/lengths to init state structure correctly
    dummy_w = np.zeros((n_r, n_r))
    dummy_l = np.zeros((n_r, n_r))
    _, state_proto = init_visual_search(hp, key, connectome_weights=dummy_w, connectome_lengths=dummy_l)
    
    return params, state_proto, hp

def create_test_stimulus(hp):
    """
    Create a simple stimulus: 128x128 with a Red Square in top-right quadrant.
    """
    img = onp.zeros((128, 128, 3), dtype=onp.float32)
    # Red Square at (96, 32) approx (x, y)
    # 0,0 is top-left.
    # Top-Right quadrant: x > 64, y < 64
    
    cx, cy = 96, 32
    r = 10
    img[cy-r:cy+r, cx-r:cx+r, 0] = 1.0 # Red channel
    
    return img

def probe_a_belief_update():
    print("\n=== Probe A: Belief Update (Empty -> Object) ===")
    params, state_proto, hp = load_trained_params()
    if params is None: return

    # Setup
    img_np = create_test_stimulus(hp)
    img = jax.device_put(img_np)[None, ...] # (1, 128, 128, 3)
    
    # Scanpath: 
    # Steps 0-4: Bottom-Left (Empty) -> (-0.5, 0.5)
    # Steps 5-9: Top-Right (Object) -> (0.5, -0.5)
    # Note: Our coords are [-1, 1]. x=0.5 -> 3/4 width (96). y=-0.5 -> 1/4 height (32).
    
    T = 10
    pos_empty = np.array([-0.5, 0.5])
    pos_obj = np.array([0.5, -0.5])
    
    scanpath = onp.zeros((1, T, 2))
    scanpath[:, :5, :] = pos_empty
    scanpath[:, 5:, :] = pos_obj
    scanpath = jax.device_put(scanpath)
    
    # Task: 0 (Color)
    tasks = np.array([0])
    
    # Run Rollout (Passive)
    # We need a rollout function.
    rollout = make_rollout(hp, n_steps=T)
    
    # Prepare State
    BATCH_SIZE = 1
    M_batch = np.repeat(state_proto.M, BATCH_SIZE, axis=0)
    history_batch = None
    if state_proto.history is not None:
        history_batch = np.repeat(state_proto.history, BATCH_SIZE, axis=1)
    lags = state_proto.lags
    state = NetworkState(M=M_batch, history=history_batch, step=0, lags=lags)
    
    logits_seq, _, _, _, _, _, _ = rollout(
        params, state, img, tasks, 'passive', scanpath, None
    )
    
    # Analyze Logits
    # Class 0: Red, 1: Green, 2: Blue
    logits_red = logits_seq[0, :, 0]
    logits_green = logits_seq[0, :, 1]
    logits_blue = logits_seq[0, :, 2]
    
    print(f"{ 'Step':<5} | {'Loc':<10} | {'Red':<8} | {'Green':<8} | {'Blue':<8} | {'Winner'}")
    print("-" * 60)
    for t in range(T):
        loc = "Empty" if t < 5 else "Object"
        r, g, b = logits_red[t], logits_green[t], logits_blue[t]
        winner = ["Red", "Green", "Blue"][np.argmax(np.array([r, g, b]))]
        print(f"{t:<5} | {loc:<10} | {r:<8.3f} | {g:<8.3f} | {b:<8.3f} | {winner}")

    # Check for update
    diff = np.mean(logits_red[5:]) - np.mean(logits_red[:5])
    print(f"\nMean Red Logit Change (Object - Empty): {diff:.4f}")
    if diff > 0.5:
        print("PASS: Significant positive shift in Red confidence when seeing object.")
    else:
        print("FAIL: No significant shift or negative shift.")

def probe_b_fef_targeting():
    print("\n=== Probe B: FEF Targeting (Saccade Vector Field) ===")
    params, state_proto, hp = load_trained_params()
    if params is None: return

    # Setup: Center Red Object
    img = onp.zeros((128, 128, 3), dtype=onp.float32)
    cx, cy = 64, 64 # Center
    img[cy-10:cy+10, cx-10:cx+10, 0] = 1.0
    img = jax.device_put(img)[None, ...] # (1, H, W, C)
    
    # Grid of positions around center
    # [-0.5, -0.5], [0.5, -0.5], ...
    # Target is at (0, 0)
    
    grid_x = [-0.5, 0.0, 0.5]
    grid_y = [-0.5, 0.0, 0.5]
    
    print(f"{ 'Pos (x,y)':<15} | {'Saccade (dx,dy)':<20} | {'Ideal':<20} | {'CosSim':<8}")
    print("-" * 75)
    
    total_cos_sim = 0.0
    count = 0
    
    # Task: Color (shouldn't matter much for saliency, but let's stick to 0)
    task_idx = np.array([0])
    
    # We want single step evaluation.
    # We need to construct a batch of positions to run efficiently or loop.
    # Let's loop for clarity.
    
    BATCH_SIZE = 1
    M_batch = np.repeat(state_proto.M, BATCH_SIZE, axis=0)
    history_batch = None
    if state_proto.history is not None:
        history_batch = np.repeat(state_proto.history, BATCH_SIZE, axis=1)
    lags = state_proto.lags
    state = NetworkState(M=M_batch, history=history_batch, step=0, lags=lags)

    for py in grid_y:
        for px in grid_x:
            if px == 0 and py == 0: continue # Skip center (on target) 
            
            pos = np.array([[px, py]])
            
            # Extract patch
            patch = extract_patches(img, pos, hp.patch_size)
            
            # Run one step
            # Note: We are testing the *initial* reaction to the scene from this position.
            # We assume state is fresh (or we should reset it? likely fresh is cleaner for vector field)
            _, (logits, saccade, _, _, _) = agent_step(
                params, state, patch, pos, task_idx, 0, hp
            )
            
            dx, dy = saccade[0, 0], saccade[0, 1]
            
            # Ideal vector: towards (0,0)
            ideal_x, ideal_y = -px, -py
            norm = np.sqrt(ideal_x**2 + ideal_y**2)
            ideal_x, ideal_y = ideal_x/norm, ideal_y/norm
            
            # Saccade norm
            s_norm = np.sqrt(dx**2 + dy**2) + 1e-8
            sx, sy = dx/s_norm, dy/s_norm
            
            # Cosine Similarity
            cos_sim = sx*ideal_x + sy*ideal_y
            total_cos_sim += cos_sim
            count += 1
            
            print(f"[{px:>4.1f}, {py:>4.1f}]   | [{dx:>5.2f}, {dy:>5.2f}]     | [{ideal_x:>5.2f}, {ideal_y:>5.2f}]     | {cos_sim:>6.2f}")

    avg_cos = total_cos_sim / count
    print(f"\nAverage Cosine Similarity: {avg_cos:.4f}")
    if avg_cos > 0.0:
        print("Assessment: FEF shows POSITIVE attraction (points towards object on average).")
    elif avg_cos > 0.5:
        print("Assessment: FEF shows STRONG attraction.")
    else:
        print("Assessment: FEF shows NEGATIVE/RANDOM attraction.")

if __name__ == "__main__":
    probe_a_belief_update()
    probe_b_fef_targeting()
