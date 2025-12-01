import jax
import jax.numpy as np
import numpy as onp
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vbjax.app.visual_search.model import init_visual_search, VisualSearchHyperparameters, IDX_R_V1, IDX_R_FEF, IDX_R_PFC
from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.visual_search.train import make_rollout

def generate_single_trial_with_metadata(seed=42, size=128):
    rng = onp.random.default_rng(seed)
    image = onp.zeros((size, size, 3), dtype=onp.float32)
    
    n_objects = rng.integers(3, 11)
    
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)] # R, G, B
    color_names = ['Red', 'Green', 'Blue']
    shapes = ['circle', 'square']
    
    counts = {'red': 0, 'green': 0, 'blue': 0, 'circle': 0, 'square': 0}
    objects = []

    for i in range(n_objects):
        shape_type = rng.choice(shapes)
        color_idx = rng.integers(0, 3)
        color = colors[color_idx]
        c_name = color_names[color_idx]
        
        # Update counts
        if color_idx == 0: counts['red'] += 1
        elif color_idx == 1: counts['green'] += 1
        elif color_idx == 2: counts['blue'] += 1
        counts[shape_type] += 1
        
        r = rng.integers(8, 16)
        cx = rng.integers(r, size - r)
        cy = rng.integers(r, size - r)
        
        obj = {
            'id': i,
            'shape': shape_type,
            'color': c_name,
            'pos': (cx, cy),
            'r': r
        }
        objects.append(obj)
        
        if shape_type == 'circle':
            y, x = onp.ogrid[:size, :size]
            mask_shape = (x - cx)**2 + (y - cy)**2 <= r**2
            image[mask_shape] = color
        else:
            x0, x1 = cx - r, cx + r
            y0, y1 = cy - r, cy + r
            image[y0:y1, x0:x1] = color
            
    # Task
    task_id = rng.integers(0, 2)
    if task_id == 0: # Max Color
        c_counts = [counts['red'], counts['green'], counts['blue']]
        label = onp.argmax(c_counts)
        task_name = "Max Frequency Color"
        class_names = ['Red', 'Green', 'Blue']
    else: # Max Shape
        s_counts = [counts['circle'], counts['square']]
        label = onp.argmax(s_counts)
        task_name = "Max Frequency Shape"
        class_names = ['Circle', 'Square', 'N/A']

    return image, task_id, label, objects, task_name, class_names

def plot_trial(params_path="visual_search_params.pkl"):
    # 1. Load Params
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    # Load Connectome
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
    
    # 2. Config
    mhsa_hp = Hyperparameters(n_regions=n_r, n_heads=8, d_k=32, d_v=32, d_model=32, steps_per_token=5)
    hp = VisualSearchHyperparameters(mhsa=mhsa_hp, patch_size=32, n_tasks=2, n_classes=3, retina_channels=(16, 32))
    
    # 3. Generate Data
    # Use a fixed seed for reproducibility of the "random" trial, or random?
    # Let's use random to see different things each run
    seed = onp.random.randint(0, 10000)
    print(f"Generating trial with seed: {seed}")
    img_np, task_id, label, objects, task_name, class_names = generate_single_trial_with_metadata(seed=seed)
    
    images = jax.device_put(img_np[None, ...]) / 255.0
    tasks = jax.device_put(np.array([task_id]))
    
    # 4. Init State
    key = jax.random.PRNGKey(0)
    _, state_proto = init_visual_search(hp, key, connectome_weights=weights, connectome_lengths=lengths)
    
    # 5. Run Rollout
    # Need to handle n_steps from loaded params? 
    # Actually n_steps is an arg to make_rollout. The training used 30.
    rollout = make_rollout(hp, n_steps=30)
    
    logits_seq, saccades_seq, pos_seq, log_probs_seq, values_seq, surprise_seq, priority_seq = rollout(
        params, state_proto, images, tasks, mode='active', scanpaths=None, key=key
    )
    
    # 6. Analyze & Log
    pos = pos_seq[0] # (T, 2)
    logits = logits_seq[0]
    probs = jax.nn.softmax(logits, axis=-1)
    preds = np.argmax(logits, axis=-1)
    
    H, W, C = img_np.shape
    
    print(f"\n=== Trial Analysis (Task: {task_name}) ===")
    print(f"Ground Truth Label: {class_names[label]} (Idx {label})")
    print(f"{'Step':<5} | {'Pos (x,y)':<15} | {'Fixated Object':<25} | {'Prediction':<15} | {'Confidence':<10}")
    print("-" * 80)
    
    fixated_objs = set()
    
    for t in range(pos.shape[0]):
        # Pos [-1, 1] -> [0, W]
        px = (pos[t, 0] + 1) / 2 * W
        py = (pos[t, 1] + 1) / 2 * H
        
        fixated = []
        for obj in objects:
            ocx, ocy = obj['pos']
            r = obj['r']
            dist = onp.sqrt((px - ocx)**2 + (py - ocy)**2)
            # Expanded radius for "foveal attention" (patch size is 32, so radius 16)
            # If patch overlaps object significantly.
            # Patch radius ~16. Object radius ~8-16.
            if dist < (r + 16): 
                fixated.append(f"{obj['color']} {obj['shape']}")
                fixated_objs.add(obj['id'])
        
        fix_str = ", ".join(fixated) if fixated else "Background"
        
        pred_idx = preds[t]
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        conf = probs[t, pred_idx]
        
        print(f"{t:<5} | {px:>6.1f}, {py:>6.1f}  | {fix_str:<25} | {pred_name:<15} | {conf:.2f}")
        
    final_pred = preds[-1]
    is_correct = (final_pred == label)
    print("-" * 80)
    print(f"Total Objects: {len(objects)}")
    print(f"Unique Objects Fixated: {len(fixated_objs)}")
    print(f"Final Prediction: {class_names[final_pred]}")
    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
    
    # 7. Plot
    img = img_np
    vals = values_seq[0]
    surprise = surprise_seq[0]
    
    pos_px = (pos + 1) / 2 * np.array([W, H])
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)
    
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[2, :])
    
    ax00.imshow(img.astype(int)) # Ensure valid type for imshow if 0-255
    ax00.plot(pos_px[:, 0], pos_px[:, 1], 'w-o', alpha=0.7, linewidth=1.5)
    ax00.set_title(f"Scanpath (Task: {task_name})")
    ax00.plot(pos_px[0, 0], pos_px[0, 1], 'gx', markersize=10, label='Start')
    ax00.plot(pos_px[-1, 0], pos_px[-1, 1], 'rx', markersize=10, label='End')
    
    # Mark objects
    for obj in objects:
        ox, oy = obj['pos']
        ax00.text(ox, oy, str(obj['id']), color='white', ha='center', va='center', fontweight='bold')

    ax00.legend()
    
    # Value
    ax01.plot(vals, 'b-')
    ax01.set_title("Value Estimate (PFC) over Time")
    ax01.grid(True)
    
    # Probs
    ax10.plot(probs[:, 0], 'r-', label=class_names[0])
    ax10.plot(probs[:, 1], 'g-', label=class_names[1])
    if len(class_names) > 2:
        ax10.plot(probs[:, 2], 'b-', label=class_names[2])
    ax10.set_title("Class Probabilities")
    ax10.legend()
    ax10.grid(True)
    
    # Eye
    ax11.plot(pos[:, 0], label='X')
    ax11.plot(pos[:, 1], label='Y')
    ax11.set_title("Eye Position")
    ax11.set_ylim([-1.1, 1.1])
    ax11.legend()
    ax11.grid(True)
    
    # Surprise
    surprise_mean = np.mean(surprise, axis=-1)
    roi_indices = {'V1': IDX_R_V1, 'FEF': IDX_R_FEF, 'PFC': IDX_R_PFC}
    colors = {'V1': 'r', 'FEF': 'g', 'PFC': 'b'}
    
    for n in range(n_r):
        if n not in roi_indices.values():
            ax2.plot(surprise_mean[:, n], color='gray', alpha=0.1)
    for name, idx in roi_indices.items():
        ax2.plot(surprise_mean[:, idx], color=colors[name], alpha=0.9, linewidth=2.0, label=name)
            
    ax2.set_title("Neural Surprise")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("visual_search_trial.png")
    print("Saved visualization to visual_search_trial.png")

if __name__ == "__main__":
    plot_trial()