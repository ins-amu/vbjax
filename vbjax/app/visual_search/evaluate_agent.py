import jax
import jax.numpy as np
import numpy as onp
import pickle
import os
import argparse
from tqdm import tqdm

from vbjax.app.visual_search.model import init_visual_search, VisualSearchHyperparameters, NetworkState
from vbjax.ct_mhsa import Hyperparameters
from vbjax.app.visual_search.data import generate_dataset, make_scanpaths
from vbjax.app.visual_search.train import make_rollout, get_target_coords
from vbjax.app.visual_search.assessment_probes import load_trained_params

def evaluate_visual_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=500) # Several hundred trials
    parser.add_argument("--n_steps", type=int, default=30) # Max steps per episode
    parser.add_argument("--checkpoint", type=str, default="visual_search_params.pkl")
    args = parser.parse_args()

    # Load Trained Parameters
    print(f"Loading trained parameters from {args.checkpoint}...")
    params, state_proto, hp = load_trained_params(args.checkpoint)
    if params is None: return

    # Rollout function for evaluation
    rollout_fn = make_rollout(hp, n_steps=args.n_steps, deterministic=True)

    # Data Generation (fresh test set)
    print(f"Generating {args.num_episodes} test episodes...")
    test_images_np, test_tasks_np, test_labels_np, test_masks_np = generate_dataset(n_samples=args.num_episodes, seed=999) # Use different seed

    test_images = jax.device_put(test_images_np) / 255.0
    test_tasks = jax.device_put(test_tasks_np.flatten())
    test_labels = jax.device_put(test_labels_np)
    test_masks = jax.device_put(test_masks_np)

    all_accuracies = []
    all_saccade_counts = []
    all_final_dists = []
    all_scanpaths = []  # To store for plotting later
    all_correct_trials = 0
    all_incorrect_trials = 0

    print("Starting evaluation...")
    # Iterate over episodes (batch size 1 for evaluation clarity)
    key = jax.random.PRNGKey(1234)
    
    # Store data for plotting sample episodes
    sample_episode_data = []
    num_samples_to_plot = 5

    for i in tqdm(range(args.num_episodes), desc="Evaluating episodes"):
        key, episode_key = jax.random.split(key)

        img_i = test_images[i:i+1]
        task_i = test_tasks[i:i+1]
        label_i = test_labels[i:i+1]
        mask_i = test_masks[i:i+1]
        target_coords_i = get_target_coords(mask_i)

        # Prepare initial state for a single episode
        # Ensure M and history are reset for each episode
        init_M = np.zeros_like(state_proto.M) # Reset memory
        init_history = None
        if state_proto.history is not None:
            init_history = np.zeros_like(state_proto.history) # Reset history
        
        init_state_i = NetworkState(M=init_M, history=init_history, step=0, lags=state_proto.lags)

        # Run rollout in active mode
        logits_seq, saccades_seq, pos_seq, _, values_seq, surprise_seq, priority_seq = rollout_fn(
            params, init_state_i, img_i, task_i, 'active', None, episode_key
        )

        # Calculate metrics for this episode
        final_logits = logits_seq[0, -1, :]
        pred_label = np.argmax(final_logits)
        accuracy = (pred_label == label_i[0]).astype(np.float32)
        all_accuracies.append(accuracy)

        if accuracy:
            all_correct_trials += 1
        else:
            all_incorrect_trials += 1

        # Saccade count (non-redundant moves, or just n_steps if always moving)
        # For simplicity, count all steps for now (max_steps)
        saccade_count = args.n_steps # Max steps, agent always moves.
        all_saccade_counts.append(saccade_count)

        # Final distance to target
        final_pos = pos_seq[0, -1, :]
        final_dist = np.sqrt(np.sum((final_pos - target_coords_i[0])**2))
        all_final_dists.append(final_dist)
        
        # Store scanpath
        all_scanpaths.append(pos_seq[0]) # (N_STEPS, 2)

        # Store detailed data for sample episodes
        if i < num_samples_to_plot:
            sample_episode_data.append({
                "episode_idx": i,
                "image": test_images_np[i],
                "task": test_tasks_np[i],
                "label": test_labels_np[i],
                "mask": test_masks_np[i],
                "target_coords": target_coords_i[0],
                "logits_seq": logits_seq[0],
                "saccades_seq": saccades_seq[0],
                "pos_seq": pos_seq[0],
                "values_seq": values_seq[0],
                "surprise_seq": surprise_seq[0],
                "priority_seq": priority_seq[0],
                "accuracy": accuracy
            })

    # Aggregate Results
    avg_accuracy = np.mean(np.array(all_accuracies))
    std_accuracy = np.std(np.array(all_accuracies))
    avg_saccades = np.mean(np.array(all_saccade_counts))
    avg_final_dist = np.mean(np.array(all_final_dists))
    std_final_dist = np.std(np.array(all_final_dists))

    print("\n=== Evaluation Results ===")
    print(f"Total Episodes: {args.num_episodes}")
    print(f"Correct Trials: {all_correct_trials}")
    print(f"Incorrect Trials: {all_incorrect_trials}")
    print(f"Average Accuracy: {avg_accuracy:.2%} (Std: {std_accuracy:.2%})")
    print(f"Average Saccades per Episode: {avg_saccades:.1f}")
    print(f"Average Final Distance to Target: {avg_final_dist:.4f} (Std: {std_final_dist:.4f})")

    # Save scanpaths and other data for plotting
    eval_data = {
        "accuracies": all_accuracies,
        "saccade_counts": all_saccade_counts,
        "final_dists": all_final_dists,
        "scanpaths": all_scanpaths,
        "test_images": test_images_np,
        "test_tasks": test_tasks_np,
        "test_labels": test_labels_np,
        "sample_episode_data": sample_episode_data
    }
    with open("evaluation_results.pkl", "wb") as f:
        pickle.dump(eval_data, f)
    print("Evaluation results saved to evaluation_results.pkl")

    print("\n--- To plot sample episodes, run the following Python code locally (requires matplotlib): ---")
    print("import pickle")
    print("import matplotlib.pyplot as plt")
    print("import jax.numpy as np")
    print("\nwith open('evaluation_results.pkl', 'rb') as f:")
    print("    eval_data = pickle.load(f)")
    print("\nfor i, episode_data in enumerate(eval_data['sample_episode_data']):")
    print("    if i >= 5: break # Plot first 5 samples")
    print("    img = episode_data['image']")
    print("    mask = episode_data['mask']")
    print("    scanpath = episode_data['pos_seq']")
    print("    target_coords = episode_data['target_coords']")
    print("    logits_seq = episode_data['logits_seq']")
    print("    task_name = {0: 'Color', 1: 'Shape'}[episode_data['task']] # Assuming task 0 or 1")
    print("    label_name = {0: 'Red', 1: 'Green', 2: 'Blue'}[episode_data['label']] # Assuming 3 classes")
    print("    accuracy = 'Correct' if episode_data['accuracy'] == 1.0 else 'Incorrect'")
    
    print("\n    fig, axes = plt.subplots(1, 2, figsize=(12, 6))")
    
    print("    # Plot Image and Scanpath")
    print("    axes[0].imshow(img.astype(np.uint8))")
    print("    axes[0].imshow(mask, alpha=0.5, cmap='gray') # Overlay mask")
    print("    # Convert normalized positions [-1, 1] to pixel coordinates [0, 128]")
    print("    pixel_scanpath = (scanpath + 1) / 2 * 128")
    print("    pixel_target = (target_coords + 1) / 2 * 128")
    print("    axes[0].plot(pixel_scanpath[:, 0], pixel_scanpath[:, 1], 'o-', color='yellow', alpha=0.7, markersize=4, linewidth=2)")
    print("    axes[0].plot(pixel_target[0], pixel_target[1], 'x', color='cyan', markersize=10, linewidth=3, label='Target COM')")
    print("    axes[0].set_title(f'Episode {i} ({accuracy}) | Task: {task_name}, Label: {label_name}')")
    print("    axes[0].legend()")

    print("    # Plot Logits over time")
    print("    for c in range(logits_seq.shape[-1]):")
    print("        axes[1].plot(logits_seq[:, c], label=f'Class {c}')")
    print("    axes[1].axhline(y=0, color='gray', linestyle='--')")
    print("    axes[1].set_title('Logits over Time')")
    print("    axes[1].set_xlabel('Time Step')")
    print("    axes[1].set_ylabel('Logit Value')")
    print("    axes[1].legend()")
    
    print("    plt.tight_layout()")
    print("    plt.show()")

if __name__ == "__main__":
    evaluate_visual_search()
