import jax.numpy as np
import matplotlib.pyplot as plt
import jax 


def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    return plt.loglog(f, np.abs(np.fft.rfft(s)))[0]

def noise_psd(shape, key, sigma=1, psd = lambda f: 1, **kwargs):
        X_white = np.fft.rfft(jax.random.normal(key, shape)*sigma)
        S = psd(np.fft.rfftfreq(shape[1]), **kwargs)
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def PSDGenerator(f, *args, **kwargs):
    return lambda shape, *args, **kwargs,: noise_psd(shape, *args, **kwargs, psd=f)

@PSDGenerator
def white_noise(f):
    return 1

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    return f

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

@PSDGenerator
def spectral_exponent(f, exponent=1):
    return 1/np.where(f == 0, float('inf'), f**exponent)


