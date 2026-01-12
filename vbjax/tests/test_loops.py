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


def test_sdde_zero_delay():
    """Test SDDE with zero delay (nh=0) should behave like regular SDE"""
    dt = 0.01
    nh = 0  # zero delay
    t_max = 0.5
    n_steps = int(t_max / dt)
    
    # Simple ODE: dx/dt = -x
    def dfun(xt, x, t, p):
        return -x  # no delay, just -x
    
    # No noise
    gfun = 0.0
    
    # Initial condition
    x0 = 1.0
    buf_size = nh + 1 + n_steps
    buf = np.ones(buf_size) * x0
    
    # Create SDDE integrator
    step, loop = vb.make_sdde(dt, nh, dfun, gfun)
    
    # Integrate
    final_buf, trajectory = loop(buf, None)
    
    # Basic sanity checks
    assert trajectory.shape == (n_steps,), f"Expected shape ({n_steps},), got {trajectory.shape}"
    assert np.all(np.isfinite(trajectory)), "Solution contains non-finite values"
    
    # Compare with analytical solution x(t) = exp(-t)
    ts = np.arange(0, t_max, dt)
    analytical = np.exp(-ts)
    
    # Should match very closely now that we use optimized SDE integrator for zero delay
    max_error = np.max(np.abs(trajectory[:-1] - analytical[1:]))
    assert max_error < 1e-4, f"Max error {max_error:.2e} too large for zero-delay case"


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
    assert xs.x.shape == z.x[:-(nh+1)].shape

    # TODO test also w/ gfun generating pytree
    def g(x, p):
        return State(x=0.01, y=0.02)
    _, loop = vb.make_sdde(dt, nh, f, g)
    buf = State(x=vb.randn(100, 32), y=vb.randn(100, 32))
    _, xs = loop(buf, None)
    assert xs.x.shape == buf.x[:-(nh+1)].shape
    assert xs.y.shape == buf.y[:-(nh+1)].shape


def test_continuation():
    import jax, jax.numpy as jp, vbjax as vb
    import numpy
    
    def f(buf, x, t, p):
        return vb.mpr_dfun(x, (0,0), p)
    
    nh = 16
    dt = 0.1
    clen = 30
    buf = jp.zeros((nh + 1 + clen, 2, 1)) + 1.0
    p = vb.mpr_default_theta._replace(eta=-1.0)._replace(tau=3.0)
    
    # run it
    _, loop = vb.make_sdde(dt, nh, f, 0.0)
    cc = vb.make_continuation(loop, clen, nh, 1, 1, stochastic=False)
    # b, xs = jax.lax.scan(lambda buf,key: cc(buf,p,key), buf, vb.keys[:3])
    xs = []
    for i in range(3):
        # buf, x = loop(buf, p) # cc(buf, p, vb.keys[i])
        # buf = jp.roll(buf, -clen+1, axis=0)
        buf, x = cc(buf, p, vb.keys[i])
        xs.append(x)
    xs = np.array(xs)

    numpy.testing.assert_allclose(
        loop(jp.zeros((nh + 1 + clen*3, 2, 1)) + 1.0, p)[1][:, 1, 0],
        xs.reshape(-1, 2)[:, 1],
        1e-6, 2e-5
    )