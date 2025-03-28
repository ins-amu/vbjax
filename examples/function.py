import time
import numpy as np
import pylab as pl
import vbjax as vb
import jax
import jax.numpy as jp
import jax.example_libraries.optimizers as jopt
import tqdm
import torch
from sbi.inference import infer

# Function to perform Bayesian Regression using SBI
def quick_bayesian_regression(parameters, observations, method="NPE"):
    """
    Perform Bayesian regression using SBI framework.
    
    Parameters:
        parameters (numpy.ndarray): Input parameters.
        observations (numpy.ndarray): Corresponding observed values.
        method (str): SBI inference method. Default is "NPE" (Neural Posterior Estimation).
    
    Returns:
        posterior: Inferred posterior distribution.
    """
    parameters = torch.tensor(parameters, dtype=torch.float32)
    observations = torch.tensor(observations, dtype=torch.float32)
    
    posterior = infer(parameters, observations, method=method)
    return posterior

# First model: Single Jansen-Rit
def model(state, parameters):
    return vb.jr_dfun(state, 0, parameters)

# Function to run simulation & compute Welch PSD
def run_sim_psd(parameters, rng_key):
    A, B, a, b, v0, r, J, lsig = parameters 
    dt = 2.0  # ms (500 Hz sampling frequency)
    ntime = int(60e3 / dt)
    initial_state = jp.ones((6, 1))
    _, loop = vb.make_sde(dt=dt, dfun=model, gfun=jp.exp(lsig))
    noise = vb.randn(ntime, *initial_state.shape, key=rng_key)
    parameters = vb.jr_default_theta._replace(A=A, B=B, a=a, b=b)
    states = loop(initial_state, noise, parameters)
    lfp = states[:, 1] - states[:, 0]
    lfp = jp.diff(lfp, axis=0)  # Remove 1/f
    
    win_size = int(4e3 / dt)
    overlap = win_size // 4  # 25% overlap
    ftfreq = jp.fft.fftfreq(win_size, dt) * 1e3  # kHz -> Hz
    
    windows = jp.array([lfp[i * overlap:i * overlap + win_size, 0]
                        for i in range((len(lfp) - win_size) // overlap)])
    windows = windows * jp.hanning(win_size)
    windows_fft = jp.fft.fft(windows, axis=1)
    windows_psd = jp.mean(jp.abs(windows_fft), axis=0)
    return ftfreq, windows_psd

# Load data
Pz = np.load('Sebastien_Spectrum_Pz.npy')

# Run simulation for example parameters
parameters = jp.array([
    (3.25, 22.0, 0.1, 0.05, 5.52, 0.56, 135.0, -8.0),
    (3.25, 22.0, 0.1, 0.05, 5.52, 0.56, 135.0, -7.0),
    (3.31, 22.5, 0.11, 0.049, 5.59, 0.55, 130.0, -8.0),
])

rng_keys = jax.random.split(jax.random.PRNGKey(1106), len(parameters))
psds = [run_sim_psd(p, k) for p, k in zip(parameters, rng_keys)]

# Bayesian Regression Instead of Manual Loss Function
posterior = quick_bayesian_regression(parameters, np.array([psd[1] for psd in psds]))
print("Bayesian Inference Done!")

# Show PSD results
pl.figure()
ftfreq = psds[0][0]
ftmask = (ftfreq > 0) * (ftfreq < 80)
for _, psd in psds:
    pl.plot(ftfreq[:175], jp.log(psd[ftmask])[:175])
pl.plot(ftfreq[:175], Pz.T, 'r')
pl.xlim([0, 50])
pl.grid(True)
pl.legend([str(_) for _ in parameters])
pl.ylabel('PSD')
pl.xlabel('Hz')
pl.title('Example Simulated Welch PSD on 60s Jansen-Rit')
pl.show()
