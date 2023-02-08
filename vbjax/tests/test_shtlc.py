import jax
import jax.numpy as np

from vbjax import make_shtdiff


def test_shtdiff():
    nlat, nlon = 32, 64
    lc = make_shtdiff(nlat=32, nlon=nlon)
    key = jax.random.PRNGKey(42)
    x0 = jax.random.normal(key, (nlat, nlon))
    x1 = lc(x0)
