import numpy as np
import torch
from function import quick_bayesian_regression  

def test_bayesian_regression():
    params = np.array([[3.25, 22.0, 0.1, 0.05]])
    obs = np.array([[5.52, 0.56, 135.0, -8.0]])

    posterior = quick_bayesian_regression(params, obs)
    
    assert posterior is not None  
    assert isinstance(posterior, torch.Tensor)  # Ensure output is a tensor
    assert posterior.mean().item() > 0  # Check for a reasonable mean value
