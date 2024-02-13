import jax
import jax.numpy as np
import vbjax as vb


def test_sde():
    f = lambda x,_: -x
    g = lambda *_: 1e-2
    dt = 0.1
    _, run = vb.make_sde(dt, f, g)
    zs = vb.randn(100, 32)
    xs = run(zs[0]+1, zs, None)
    assert xs.shape == zs.shape


def test_ode():
    f = lambda x, _: -x
    dt = 0.1
    _, run = vb.make_ode(dt, f)
    x0 = np.r_[:32].astype('f')
    xs = run(x0, np.r_[:64], None)
    assert xs.shape == (64, 32)


def test_dde():
    def dfun(xt, x, t, p):
        xd = xt[0, t-100]
        dx = x - x**3/3 + p*xt[0, t-100]
        return dx

    _, loop = vb.make_dde(0.1, 100, dfun)
    xt0 = np.ones((1, 200))
    xt1,t = loop(xt0, 0.2)
    assert xt1.shape == (1, 200)


def test_sdde():
    def dfun(xt, x, t, p):
        return -xt[t - 5]
    _, sdde = vb.make_sdde(1.0, 5, dfun, 0.01)
    sdde(vb.randn(20)+10, None)


# TODO theta method? https://gist.github.com/maedoc/c47acb9d346e31017e05324ffc4582c1
    
def test_heun_pytree():
    from collections import namedtuple
    State = namedtuple('State', 'x y')
    def f(x: State, p):
        return State(x.y, -x.x)
    dt = 0.1

    # first test with ode
    _, loop = vb.make_ode(dt, f)
    x = np.ones(32)
    y = np.zeros(32)
    x0 = State(x, y)
    xs = loop(x0, np.r_[:64], None)
    assert xs.x.shape == (64, 32)
    assert xs.y.shape == (64, 32)

    # then test with sde
    _, loop = vb.make_sde(dt, f, 1e-2)
    z = State(x=vb.randn(100, 32), y=np.zeros((100, 32)))
    xs = loop(x0, z, None)
    assert xs.x.shape == z.x.shape

    # now with sdde
    def f(xs: State, x: State, t, p):
        return State(xs.y[t-3], -x.x)
    nh = 5
    _, loop = vb.make_sdde(dt, nh, f, 1e-2)
    _, xs = loop(xs, None)
    assert xs.x.shape == z.x[:-nh].shape

    # TODO test also w/ gfun generating pytree