import jax
import jax.numpy as np
import numpy as onp
import pickle
import os
from vbjax.app.visual_search.model import init_visual_search, VisualSearchHyperparameters
from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.visual_search.train import make_rollout
from vbjax.app.visual_search.plot_trial import generate_single_trial_with_metadata

def analyze_trials(n_trials=100, params_path="visual_search_params.pkl"):
    # 1. Load Params & Config
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(base_dir, 'weights.txt')
    lengths_path = os.path.join(base_dir, 'tract_lengths.txt')
    
    full_weights = onp.loadtxt(weights_path)
    full_lengths = onp.loadtxt(lengths_path)
    n_r = 38
    weights = full_weights[:n_r, :n_r] / (np.max(full_weights[:n_r, :n_r]) + 1e-8)
    lengths = full_lengths[:n_r, :n_r]
    
    weights = jax.device_put(weights)
    lengths = jax.device_put(lengths)
    
    mhsa_hp = Hyperparameters(n_regions=n_r, n_heads=8, d_k=32, d_v=32, d_model=32, steps_per_token=5)
    hp = VisualSearchHyperparameters(mhsa=mhsa_hp, patch_size=32, n_tasks=2, n_classes=3, retina_channels=(16, 32), max_steps=100)
    
    # Init State
    key = jax.random.PRNGKey(42)
    _, state_proto = init_visual_search(hp, key, connectome_weights=weights, connectome_lengths=lengths)
    
    rollout = make_rollout(hp, n_steps=30)
    rollout_jit = jax.jit(rollout, static_argnames=['mode'])
    
    print(f"Running {n_trials} trials...")
    
    results = []
    
    for i in range(n_trials):
        # Generate Data
        img_np, task_id, label, objects, task_name, class_names = generate_single_trial_with_metadata(seed=i)
        
        images = jax.device_put(img_np[None, ...]) / 255.0
        tasks = jax.device_put(np.array([task_id]))
        
        # Run
        key, subkey = jax.random.split(key)
        logits_seq, saccades_seq, pos_seq, _, _, _, _ = rollout_jit(
            params, state_proto, images, tasks, mode='active', scanpaths=None, key=subkey
        )
        
        # Analyze
        logits = logits_seq[0] # (T, 3)
        probs = jax.nn.softmax(logits, axis=-1)
        preds = np.argmax(logits, axis=-1)
        pos = pos_seq[0]
        
        final_pred = preds[-1]
        is_correct = (final_pred == label)
        
        # Confidence
        final_conf = probs[-1, final_pred]
        mean_conf = np.mean(np.max(probs, axis=-1))
        
        # Stability
        # Step where prediction became final_pred and stayed there
        stable_step = 0
        for t in range(len(preds)-1, -1, -1):
            if preds[t] != final_pred:
                stable_step = t + 1
                break
        
        # Coverage
        fixated_objs = set()
        H, W, C = img_np.shape
        for t in range(pos.shape[0]):
            px = (pos[t, 0] + 1) / 2 * W
            py = (pos[t, 1] + 1) / 2 * H
            for obj in objects:
                ox, oy = obj['pos']
                dist = onp.sqrt((px - ox)**2 + (py - oy)**2)
                if dist < (obj['r'] + 16):
                    fixated_objs.add(obj['id'])
                    
        coverage = len(fixated_objs) / len(objects)
        
        results.append({
            'id': i,
            'task': task_name,
            'correct': is_correct,
            'final_conf': final_conf,
            'mean_conf': mean_conf,
            'stable_step': stable_step,
            'coverage': coverage,
            'n_objects': len(objects)
        })
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{n_trials}...")
            
        if i < 5:
            print(f"Trial {i}: Preds={preds} (Label={label}) | Correct={is_correct}")

    # Aggregate
    df_correct = [r for r in results if r['correct']]
    df_incorrect = [r for r in results if not r['correct']]
    
    acc = len(df_correct) / n_trials
    
    print("\n=== Analysis Summary ===")
    print(f"Total Trials: {n_trials}")
    print(f"Accuracy: {acc:.2%}")
    
    # Task Breakdown
    tasks = set(r['task'] for r in results)
    for t in tasks:
        t_res = [r for r in results if r['task'] == t]
        t_acc = len([r for r in t_res if r['correct']]) / len(t_res)
        print(f"  - {t}: {t_acc:.2%} ({len(t_res)} trials)")

    print("\nMetrics (Avg):")
    print(f"  {'Metric':<20} | {'All':<10} | {'Correct':<10} | {'Incorrect':<10}")
    print("-" * 60)
    
    def get_avg(data, key):
        if not data: return 0.0
        return float(np.mean(np.array([d[key] for d in data])))
        
    metrics = ['final_conf', 'mean_conf', 'stable_step', 'coverage', 'n_objects']
    for m in metrics:
        v_all = get_avg(results, m)
        v_cor = get_avg(df_correct, m)
        v_inc = get_avg(df_incorrect, m)
        print(f"  {m:<20} | {v_all:<10.2f} | {v_cor:<10.2f} | {v_inc:<10.2f}")

    print("\nPathologies:")
    # Always Low Confidence?
    low_conf_count = len([r for r in results if r['mean_conf'] < 0.6])
    print(f"  - Low Mean Confidence (<0.6): {low_conf_count} trials")
    
    # Immediate Decision?
    immediate = len([r for r in results if r['stable_step'] == 0])
    print(f"  - Immediate Decision (Step 0): {immediate} trials")

if __name__ == "__main__":
    analyze_trials()
