import jax
import jax.numpy as np
import numpy as onp
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vbjax.app.visual_search.model import init_visual_search, VisualSearchHyperparameters, IDX_R_V1, IDX_R_FEF, IDX_R_PFC
from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.visual_search.data import generate_dataset
from vbjax.app.visual_search.train import make_rollout

def plot_trial(params_path="visual_search_params.pkl"):
    # 1. Load Params
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    # Load Connectome (needed for init state shape)
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(base_dir, 'weights.txt')
    lengths_path = os.path.join(base_dir, 'tract_lengths.txt')
    
    full_weights = onp.loadtxt(weights_path)
    full_lengths = onp.loadtxt(lengths_path)
    
    n_r = 38
    weights = full_weights[:n_r, :n_r]
    lengths = full_lengths[:n_r, :n_r]
    
    weights = weights / (np.max(weights) + 1e-8)
    weights = jax.device_put(weights)
    lengths = jax.device_put(lengths)
    
    # 2. Setup Config (Must match training)
    mhsa_hp = Hyperparameters(
        n_regions=n_r, 
        n_heads=8,   
        d_k=32, d_v=32, d_model=32, 
        steps_per_token=5
    )
    hp = VisualSearchHyperparameters(
        mhsa=mhsa_hp,
        patch_size=32, 
        n_tasks=2,
        n_classes=3,
        retina_channels=(16, 32) 
    )
    
    # 3. Generate 1 Sample
    images_np, tasks_np, labels_np, masks_np = generate_dataset(n_samples=1)
    
    images = jax.device_put(images_np) / 255.0
    tasks = jax.device_put(tasks_np.flatten())
    
    # 4. Init State
    key = jax.random.PRNGKey(0)
    _, state_proto = init_visual_search(hp, key, connectome_weights=weights, connectome_lengths=lengths)
    
    # 5. Run Rollout
    rollout = make_rollout(hp)
    
    # Active Mode
    logits_seq, saccades_seq, pos_seq, log_probs_seq, values_seq, surprise_seq = rollout(
        params, state_proto, images, tasks, mode='active', scanpaths=None, key=key
    )
    
    # 6. Visualize
    img = images_np[0]
    pos = pos_seq[0] # (T, 2)
    vals = values_seq[0] # (T,)
    logits = logits_seq[0] # (T, 3)
    probs = jax.nn.softmax(logits, axis=-1)
    surprise = surprise_seq[0] # (T*K, N, H)
    
    T = pos.shape[0]
    H, W, C = img.shape
    
    # Convert pos [-1, 1] to pixels
    pos_px = (pos + 1) / 2 * np.array([W, H])
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)
    
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[2, :])
    
    # A. Image with Scanpath
    ax00.imshow(img)
    ax00.plot(pos_px[:, 0], pos_px[:, 1], 'r-o', alpha=0.7)
    ax00.set_title(f"Scanpath (Label: {labels_np[0]})")
    ax00.plot(pos_px[0, 0], pos_px[0, 1], 'gx', markersize=10, label='Start')
    ax00.plot(pos_px[-1, 0], pos_px[-1, 1], 'bx', markersize=10, label='End')
    ax00.legend()
    
    # B. Value Estimate
    ax01.plot(vals, 'b-')
    ax01.set_title("Value Estimate (PFC) over Time")
    ax01.set_xlabel("Step")
    ax01.set_ylabel("Value")
    ax01.grid(True)
    
    # C. Class Probabilities
    ax10.plot(probs[:, 0], 'r-', label='Class 0')
    ax10.plot(probs[:, 1], 'g-', label='Class 1')
    ax10.plot(probs[:, 2], 'b-', label='Class 2')
    ax10.set_title("Class Probabilities (PFC) over Time")
    ax10.set_xlabel("Step")
    ax10.set_ylabel("Probability")
    ax10.legend()
    ax10.grid(True)
    
    # D. Eye Position (X/Y)
    ax11.plot(pos[:, 0], label='X')
    ax11.plot(pos[:, 1], label='Y')
    ax11.set_title("Eye Position (FEF Output) over Time")
    ax11.set_xlabel("Step")
    ax11.set_ylim([-1.1, 1.1])
    ax11.legend()
    ax11.grid(True)
    
    # E. Surprise Signal (Key Regions)
    # Mean over Heads for clearer regional plot? Or specific head?
    # Let's plot Mean Surprise per Region
    surprise_mean = np.mean(surprise, axis=-1) # (TotalSteps, N)
    
    # Regions of Interest
    roi_indices = {
        'V1': IDX_R_V1,
        'FEF': IDX_R_FEF,
        'PFC': IDX_R_PFC
    }
    colors = {'V1': 'r', 'FEF': 'g', 'PFC': 'b'}
    
    # Plot background (gray)
    for n in range(n_r):
        if n not in roi_indices.values():
            ax2.plot(surprise_mean[:, n], color='gray', alpha=0.1, linewidth=0.5)
            
    # Plot ROIs
    for name, idx in roi_indices.items():
        ax2.plot(surprise_mean[:, idx], color=colors[name], alpha=0.9, linewidth=2.0, label=name)
            
    ax2.set_title("Neural Surprise (Mean over Heads) - Anatomical ROIs")
    ax2.set_xlabel("Micro-Step (dt)")
    ax2.set_ylabel("Surprise")
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("visual_search_trial.png")
    print("Saved visualization to visual_search_trial.png")

if __name__ == "__main__":
    plot_trial()