import jax
import jax.numpy as np

from vbjax import make_sde, make_ode, make_dde


def test_sde():
    f = lambda x,_: -x
    g = lambda *_: 1e-2
    dt = 0.1
    _, run = make_sde(dt, f, g)

    key = jax.random.PRNGKey(42)
    zs = jax.random.normal(key, (100, 32))

    xs = run(zs[0]+1, zs, None)
    assert xs.shape == zs.shape


def test_ode():
    f = lambda x,_: -x
    dt = 0.1
    _, run = make_ode(dt, f)

    x0 = np.r_[:32].astype('f')
    xs = run(x0, np.r_[:64], None)
    assert xs.shape == (64, 32)


def test_dde():
    def dfun(xt, x, t, p):
        xd = xt[0, t-100]
        dx = x - x**3/3 + p*xt[0, t-100]
        return dx

    _, loop = make_dde(0.1, 100, dfun)
    xt0 = np.ones((1, 200))
    xt1 = loop(xt0, np.r_[:100], 0.2)
    assert xt1.shape == (1, 200)
