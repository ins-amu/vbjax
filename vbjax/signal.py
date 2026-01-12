import scipy.signal as osp_signal
import jax.numpy as jnp
try:
  from jax._src.numpy.util import _wraps
except ImportError:
  from functools import wraps as _wraps
try:
  from jax._src.numpy.util import check_arraylike
except ImportError:
  # Fallback if check_arraylike is missing or moved
  def check_arraylike(name, x):
    pass

@_wraps(osp_signal.hilbert)
def hilbert(x, N=None, axis=-1):
  check_arraylike('hilbert', x)
  x = jnp.asarray(x)
  if x.ndim > 1:
    raise NotImplementedError("x must be 1D.")
  if jnp.iscomplexobj(x):
    raise ValueError("x must be real.")
  if N is None:
    N = x.shape[axis]
  if N <= 0:
    raise ValueError("N must be positive.")

  Xf = jnp.fft.fft(x, N, axis=axis)
  if N % 2 == 0:
    h = jnp.zeros(N, Xf.dtype).at[0].set(1).at[1:N // 2].set(2).at[N // 2].set(1)
  else:
    h = jnp.zeros(N, Xf.dtype).at[0].set(1).at[1:(N+1) // 2].set(2)

  x = jnp.fft.ifft(Xf * h, axis=axis)
  return x
