import jax
import jax.numpy as np
import numpy as onp  # Use numpy for generation to avoid JAX overhead/rng complexities during setup

def generate_shapes_image(key_seed, size=128, min_objects=3, max_objects=10):
    """
    Generates a single image with random shapes and colors.
    Returns:
        image: (size, size, 3)
        metadata: dict containing counts of shapes and colors
    """
    rng = onp.random.default_rng(key_seed)
    image = onp.zeros((size, size, 3), dtype=onp.float32)
    
    n_objects = rng.integers(min_objects, max_objects + 1)
    
    # 0: Red, 1: Green, 2: Blue
    colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0)
    ]
    
    # 0: Circle, 1: Square
    shapes = ['circle', 'square']
    
    counts = {
        'red': 0, 'green': 0, 'blue': 0,
        'circle': 0, 'square': 0
    }
    
    mask_img = onp.zeros((size, size), dtype=onp.float32)

    for _ in range(n_objects):
        shape_type = rng.choice(shapes)
        color_idx = rng.integers(0, 3)
        color = colors[color_idx]
        
        # Update counts
        if color_idx == 0: counts['red'] += 1
        elif color_idx == 1: counts['green'] += 1
        elif color_idx == 2: counts['blue'] += 1
        
        counts[shape_type] += 1
        
        # Random position and size
        # Avoid edges
        r = rng.integers(8, 16) # Radius or half-width
        cx = rng.integers(r, size - r)
        cy = rng.integers(r, size - r)
        
        if shape_type == 'circle':
            # Draw circle
            y, x = onp.ogrid[:size, :size]
            mask_shape = (x - cx)**2 + (y - cy)**2 <= r**2
            image[mask_shape] = color
            mask_img[mask_shape] = 1.0
            
        else: # square
            # Draw square
            x0, x1 = cx - r, cx + r
            y0, y1 = cy - r, cy + r
            image[y0:y1, x0:x1] = color
            mask_img[y0:y1, x0:x1] = 1.0
            
    return image, counts, mask_img

def generate_dataset(n_samples=1000, seed=42):
    """
    Generates a dataset of images and labels for two tasks:
    1. Most frequent color (0: Red, 1: Green, 2: Blue)
    2. Most frequent shape (0: Circle, 1: Square) - Note: may be ambiguous if equal, we'll take max
    
    Returns:
        images: (N, 128, 128, 3)
        tasks: (N, 2) one-hot-like or indices? 
               Let's do: (N, 1) task_id (0=Color, 1=Shape)
        labels: (N,) correct class index for the assigned task
        masks: (N, 128, 128) binary object mask
    """
    images = []
    task_ids = []
    labels = []
    masks = []
    
    rng = onp.random.default_rng(seed)
    
    for i in range(n_samples):
        img, counts, msk = generate_shapes_image(seed + i)
        images.append(img)
        masks.append(msk)
        
        # Randomly assign a task
        # Task 0: Max Color
        # Task 1: Max Shape
        task_id = rng.integers(0, 2)
        task_ids.append(task_id)
        
        if task_id == 0:
            # Color task
            c_counts = [counts['red'], counts['green'], counts['blue']]
            label = onp.argmax(c_counts) # In case of tie, takes first.
        else:
            # Shape task
            s_counts = [counts['circle'], counts['square']]
            label = onp.argmax(s_counts)
            
        labels.append(label)
        
    return (
        onp.array(images),
        onp.array(task_ids),
        onp.array(labels),
        onp.array(masks)
    )

def make_scanpaths(n_samples, n_steps, seed=42):
    """
    Generate random scanpaths (sequences of x,y coordinates).
    Coords in [-1, 1].
    Returns: (N, Steps, 2)
    """
    rng = onp.random.default_rng(seed)
    return rng.uniform(-0.8, 0.8, size=(n_samples, n_steps, 2))
