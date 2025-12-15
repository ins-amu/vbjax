import jax.numpy as np
import matplotlib.pyplot as plt
import jax 


def white_noise(f): 
    return f

def blue_noise(f):
    return np.sqrt(f)

def violet_noise(f):
    return f

def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

def spectral_exponent(f, exponent=1):
    return 1/np.where(f == 0, float('inf'), f**exponent)


def make_noise_generator(psd = lambda f: 1):
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

        """
        def gen(key, shape, sigma=1, **kwargs):
            # sigma = kwargs['sigma'] or 1
            X_white = np.fft.rfft(jax.random.normal(key, shape)*sigma)
            S = psd(np.fft.rfftfreq(shape[1]), **kwargs)
            S = S / np.sqrt(np.mean(S**2))
            X_shaped = X_white * S
            return np.fft.irfft(X_shaped)
        return gen




