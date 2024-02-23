import jax.numpy as np
import matplotlib.pyplot as plt
import jax 

def noise_psd(shape, key, sigma=1, psd = lambda f: 1, **kwargs):
        """Generate noise given desired power spectrum

        Parameters
        ==========
        shape : tuple
            (n_nodes, t_steps)
        key : function
            jax.random.PRNGKey()
        sigma (optional kwarg): float
            Standard deviation, defaults to 1
        exponent (optional kwarg) : float
            Spectral exponent for 1/f noise, defaults to 1
            
        Returns
        =======
        noise_stream : (n_nodes, t_steps) array
            
        Notes
        =====
        Example usage for white noise and 1/f noise

        >>> import vbjax as vb, import jax
        >>> key = jax.random.PRNGKey(seed)

        >>> vbjax.noise_generator.white_noise((1,1000), key, sigma=2)
        
        >>> vbjax.noise_generator.spectral_exponent((84,10000), key, sigma=2, exponent=1.1)

        """

        X_white = np.fft.rfft(jax.random.normal(key, shape)*sigma)
        S = psd(np.fft.rfftfreq(shape[1]), **kwargs)
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def PSDGenerator(f):
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


