"""
These are tests of our understanding of how Jax works.  If these break
then probably things elsewhere will also break.

"""

import numpy
import jax
import jax.dlpack
import jax.test_util
import jax.numpy as np
import vbjax as vb


def test_dlpack_numpy():
    x = numpy.random.randn(10)
    jx = jax.dlpack.from_dlpack(x.__dlpack__())
    nx = numpy.from_dlpack(jx)
    numpy.testing.assert_allclose(x, nx)

def test_custom_vjp_simple():

    @jax.custom_vjp
    def foo(a, b):
        return 2*a+b, 3*a+0.5*b

    def foo_fwd(a, b):
        return foo(a, b), None

    def foo_bwd(res, g):
        g_1, g_2 = g
        return (2*g[0]+3*g[1], g[0] + 0.5*g[1])

    foo.defvjp(foo_fwd, foo_bwd)

    def bar(a, b):
        c, d = foo(a, b)
        return 2*c + 3*d

    jax.grad(bar, [0,1])(3.,4.)

    jax.test_util.check_grads(bar, (3.0, 4.0), order=1, modes=('rev',))


def test_batched_jax_norm():
    "test how to batch norms of derivatives and Jacobians"

    # this is for neural ode, so make some layers
    nn_p, nn_f = vb.make_dense_layers(3, [13])
    assert callable(nn_f)

    x = vb.randn(100, 3, 50)
    x00 = x[0,:,0]

    f = lambda x: nn_f(nn_p, x)
    assert f(x00).shape == (3, 13)  # calling w/ vector doesn't work correctly!
    assert f(x00.reshape(-1, 1))[:,0].shape == (3, )

    # let's make the shape part of the f
    f2 = lambda x: nn_f(nn_p, x.reshape(-1, 1)).reshape(-1)
    assert f2(x00).shape == (3,)
    # yess

    # this makes the Jacobian behave like we expect
    J = jax.jacfwd(f2)
    assert J(x00).shape == (3, 3)

    # then the norms should work as expected
    nf = lambda x: jax.numpy.linalg.norm(f2(x))
    nJ = lambda x: jax.numpy.linalg.norm(J(x))
    assert nf(x00).shape == nJ(x00).shape == ()

    # now vmap that over 50 time points
    bnf = jax.vmap(nf, 1)
    assert bnf(x[0,:]).shape == (50,)

    bJf = jax.vmap(nJ, 1)
    assert bJf(x[0,:]).shape == (50,)

    # and a second over 100 batch elements
    bbnf = jax.vmap(bnf, 0)
    bbJf = jax.vmap(bJf, 0)
    assert bbnf(x).shape == bbJf(x).shape == (100, 50)


def test_loop_dict():
    import jax, jax.numpy as np
    ns = {
        "int": 3,
        "float": 3.14,
        "array": np.r_[:10.0]
    }
    def op(ns, inputs):
        ns["int"] += inputs
        return ns, ns["int"]
    @jax.jit
    def loop(ns: dict, inps: np.ndarray):
        ns, _ = jax.lax.scan(op, ns, inps)
        return ns

    ns1 = loop(ns, np.r_[:3])
    ns2 = loop(ns, np.r_[2:4])
    assert ns["int"] == 3
    assert ns1["int"] == 6
    assert ns2["int"] == 8


def _dict_step(sim, t):
    x = sim['x']
    z = sim['noise'] * jax.random.normal(sim['rng'], shape=x.shape)
    f = lambda x: x - x**3/3 + sim['coupling'] * sim['weights']@x
    x = vb.heun_step(x, f, sim['dt'])
    sim['x'] = x
    eeg = sim['eeg_gain'] @ x
    return sim, eeg

def test_loop_dict2():

    # a more complicated example
    batch_size = 32
    sim = {
        'noise': 0.1,
        'coupling': 0.01,
        'dt': 0.1,
        'x': vb.randn(164, batch_size),
        'weights': vb.randn(164,164),
        'eeg_gain': vb.randn(64, 164),
        'rng': jax.random.PRNGKey(42)
    }

    def loop(sim, ts):
        sim, eegs = jax.lax.scan(_dict_step, sim, ts)
        return eegs

    params = {
        'dt': 0.01,
        'coupling': 0.01
    }
    ts = np.r_[:100]

    def loss(params, sim, ts):
        sim = sim.copy()
        sim.update(params)
        eegs = loop(sim, ts)
        return np.sum(np.square(eegs))

    jloss = jax.jit(jax.value_and_grad(loss))

    v, grads = jloss(params, sim, ts)
    for key, g_key in grads.items():
        assert np.abs(g_key) > 0


def test_loop_dict_vmap():

    sim = {
        'noise': 0.1,
        'coupling': 0.01,
        'dt': 0.1,
        'x': vb.randn(164),
        'weights': vb.randn(164,164),
        'eeg_gain': vb.randn(64, 164),
        'rng': jax.random.PRNGKey(42)
    }

    def loop(sim, ts):
        sim, eegs = jax.lax.scan(_dict_step, sim, ts)
        return eegs

    # to vmap we need specific args to vmap over
    # in simple case this is an array like initial conditions,
    def loop_x0(x0, sim, ts):
        sim = sim.copy()
        sim['x'] = x0
        return loop(sim, ts)

    ts = np.r_[:100]

    l1 = jax.jit(jax.vmap(lambda x: loop_x0(x, sim, ts)))
    l2 = jax.jit(lambda x: loop_x0(x, sim, ts))
    l3 = jax.jit(jax.vmap(lambda x: loop_x0(x, sim, ts), in_axes=1, out_axes=-1))

    a1 = vb.randn(32, 164)
    a2 = vb.randn(164, 32)

    assert l1(a1).shape == (32, ts.size, 64)
    assert l2(a2).shape == (ts.size, 64, 32)
    assert l3(a2).shape == (ts.size, 64, 32)