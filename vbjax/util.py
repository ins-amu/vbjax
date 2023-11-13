import numpy
import jax
import jax.dlpack


def to_np(x: jax.numpy.ndarray) -> numpy.ndarray:
    if jax.default_backend() == 'gpu':
        return numpy.array(x)
    return numpy.from_dlpack(x)

def to_jax(x: numpy.ndarray):
    "Move NumPy array to JAX via DLPack."
    x_dlp = x.__dlpack__()
    x_jax: jax.numpy.ndarray = jax.dlpack.from_dlpack(x_dlp)
    # if jax.default_backend() == 'gpu':
    #     x_jax = jax.device_put(x_jax)
    return x_jax