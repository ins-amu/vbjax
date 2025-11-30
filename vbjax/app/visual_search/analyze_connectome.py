import numpy as np
import json
import os

def analyze_connectome():
    base_dir = os.path.dirname(__file__)
    centres_path = os.path.join(base_dir, 'centres.txt')
    weights_path = os.path.join(base_dir, 'weights.txt')
    lengths_path = os.path.join(base_dir, 'tract_lengths.txt')
    json_path = os.path.join(base_dir, 'regions76.json')

    # 1. Read Centres
    with open(centres_path, 'r') as f:
        lines = f.readlines()
    
    region_names = [line.strip().split()[0] for line in lines]
    n_regions = len(region_names)
    print(f"Total regions found: {n_regions}")
    
    # 2. Identify Hemisphere Split
    r_regions = [r for r in region_names if r.startswith('r')]
    l_regions = [r for r in region_names if r.startswith('l')]
    print(f"Right hemisphere regions: {len(r_regions)}")
    print(f"Left hemisphere regions: {len(l_regions)}")
    
    # Assume Right is first (based on file inspection)
    assert region_names[:len(r_regions)] == r_regions
    
    # 3. Load Matrices
    weights = np.loadtxt(weights_path)
    lengths = np.loadtxt(lengths_path)
    
    print(f"Weights shape: {weights.shape}")
    print(f"Lengths shape: {lengths.shape}")
    
    assert weights.shape == (n_regions, n_regions)
    assert lengths.shape == (n_regions, n_regions)
    
    # 4. Slice for Right Hemisphere
    n_r = len(r_regions)
    weights_r = weights[:n_r, :n_r]
    lengths_r = lengths[:n_r, :n_r]
    
    print(f"Sliced Right Hemisphere Weights shape: {weights_r.shape}")
    
    # 5. Map Functions
    with open(json_path, 'r') as f:
        region_meta = json.load(f)
        
    functional_regions = region_meta['regions']
    
    print("\n--- Functional Region Mapping (Right Hemisphere) ---")
    found_indices = {}
    
    for item in functional_regions:
        abbr = item['Abbreviation']
        role = item['Role']
        target_name = 'r' + abbr
        
        try:
            idx = r_regions.index(target_name)
            found_indices[abbr] = idx
            print(f"Found {abbr} ({role}) at index {idx} (Name: {target_name})")
        except ValueError:
            print(f"ERROR: Could not find region {target_name} in connectome!")
            
    # 6. Check V1, FEF, PFC availability
    print("\n--- Summary for Implementation ---")
    if 'V1' in found_indices:
        print(f"V1 Input Index: {found_indices['V1']}")
    else:
        print("CRITICAL: V1 not found.")
        
    if 'FEF' in found_indices:
        print(f"FEF Output Index: {found_indices['FEF']}")
    else:
        print("CRITICAL: FEF not found.")
        
    # PFC/Frontal Group
    pfc_keys = [k for k in found_indices.keys() if 'PFC' in k]
    if pfc_keys:
        indices = [found_indices[k] for k in pfc_keys]
        print(f"PFC/Frontal Indices: {indices} (Regions: {pfc_keys})")
    else:
        print("CRITICAL: No PFC regions found.")

if __name__ == "__main__":
    analyze_connectome()
