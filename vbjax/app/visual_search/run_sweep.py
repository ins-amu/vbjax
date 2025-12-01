import subprocess
import os

def run_sweep():
    base_cmd = [
        "python3", "vbjax/app/visual_search/train.py", 
        "--train_steps", "15000", 
        "--switch_step", "5000", 
        "--n_steps", "30",
        "--lr", "1e-4",
        "--aux_weight", "1.0" # Anneals from 1.0 to 0.0
    ]
    
    term_rewards = [5.0, 10.0]
    shape_rewards = [2.0, 5.0]
    
    results = []
    
    for term in term_rewards:
        for shape in shape_rewards:
            print(f"\n--- Starting Run: Term={term}, Shape={shape} ---")
            cmd = base_cmd + ["--terminal_reward", str(term), "--shaping_reward", str(shape)]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                output = result.stdout
                
                # Parse last line
                lines = output.splitlines()
                final_stats = [line for line in lines if "Step 14500" in line]
                
                if final_stats:
                    stat_line = final_stats[-1]
                    print(f"Result: {stat_line}")
                    results.append({'term': term, 'shape': shape, 'stats': stat_line})
                else:
                    print("Warning: Could not find final stats.")
                    
            except subprocess.CalledProcessError as e:
                print(f"Run Failed: {e}")
                print(e.stderr)
                
    print("\n=== Sweep Summary ===")
    for res in results:
        print(f"Term={res['term']}, Shape={res['shape']} -> {res['stats']}")

if __name__ == "__main__":
    run_sweep()