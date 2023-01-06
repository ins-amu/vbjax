import jax
import jax.numpy as np

from nfjax import make_sde, make_ode


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
