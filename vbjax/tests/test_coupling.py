from numpy.testing import assert_allclose
import jax.numpy as np
from vbjax import make_diff_cfun, make_linear_cfun


def test_linear_cfun():
    x = np.r_[0.0:1.0:32j]
    SC = np.r_[0.0:1.0:1024j].reshape((32, 32))
    cf = make_linear_cfun(SC, a=0.5, b=0.5)
    gx = cf(x)
    assert_allclose(gx, 0.5*SC@x+0.5, 1e-5, 1e-5)


def test_diff_cfun():
    x = np.r_[0.0:1.0:32j]
    SC = np.r_[0.0:1.0:1024j].reshape((32, 32))
    def cfun(x):
        xj = x              # receiving node i
        xi = x[:, None]     # sending node j, perhaps shape (nn, nn) due to delays
        sx = SC * (xj - xi) # weight each connection by SC
        gx = np.sum(sx, axis=1)  # sum over j
        return gx
    c1 = cfun(x)
    c2 = make_diff_cfun(SC)(x)
    assert_allclose(c1, c2, 1e-5, 1e-5)

