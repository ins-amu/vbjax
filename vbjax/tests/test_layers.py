import numpy as np
import vbjax as vb
import jax
import jax.numpy as jp


def test_make_dense_layers():
    """Test make_dense_layers."""
    for in_dim, extra_in, ld in [(10, 0, [10]), (10, 5, [10, 10])]:
        p, f = vb.make_dense_layers(in_dim, ld, extra_in=extra_in)
        x = np.random.randn(in_dim + extra_in, 1)
        y = f(p, x)
        assert y.shape == (in_dim, 1)


def test_node_vjp1():
    """Tests how to write the vjp (vector-Jacobian product)
    a neural ODE by hand.  It's not hard, since the main rule
    required to understand it is the "backward mode" sensitivity
    rule for matrix products: if

        C = A B

    then, denoting dloss/dA as gA and transpose as A.T,

        gA = gC @ B.T
        gB = A.T @ gC

    so for a single neural network layer whose forward computation
    is

        y = W @ x + b

    then the vjp is

        gW = gy @ x.T
        gx = W.T @ gC
        gb = gy

    The code here tests this in a custom vjp function, against
    Jax' autodiff.

    """

    a = 0.1
    (w, b), _ = vb.make_dense_layers(12, [4, 5, 6])
    # cache = [jp.zeros((1, _.shape[1])) for _ in w[:-1]]

    def fwd(args):
        w, b, x = args
        # on fwd pass, we cache the activations
        # since they are required for vjp
        cache = []
        for i in range(len(w) - 1):
            cache.append(x)
            x = w[i] @ x + b[i]
            x = jp.where(x >= 0, x, a*x)  # leaky_relu
        cache.append(x)
        x = w[-1] @ x + b[-1]
        return x, cache

    args = w, b, jp.ones((12, 1))
    y, cache = fwd(args)

    def loss(y):
        return jp.sum(y**2)

    # compute reference gradients with jax
    g_w, g_b, g_x = jax.grad(lambda a: loss(fwd(a)[0]))(args)

    # my version has to start with vjp of loss
    def my_vjp_loss(y):
        return 2*y

    # then the vjp of mlp
    def my_vjp_mlp(args, cache, g_x, g_w, g_b):
        w, b, x = args

        g_w[-1] = g_w[-1] + g_x @ cache[-1].T
        g_b[-1] = g_b[-1] + g_x
        g_x = w[-1].T @ g_x

        # work backwards through layers
        for i in range(len(w)-2, -1, -1):
            # leaky relu activation
            g_x = jp.where(cache[i+1] >= 0, g_x, a*g_x)
            # x = w@x+b
            # g_w & g_b accumulate
            g_w[i] = g_w[i] + g_x @ cache[i].T
            g_b[i] = g_b[i] + g_x
            g_x = w[i].T @ g_x
        return g_w, g_x, g_b

    gh_x = my_vjp_loss(y)
    gh_w = [jp.zeros_like(_) for _ in w]
    gh_b = [jp.zeros_like(_) for _ in b]
    gh_w, gh_x, gh_b = my_vjp_mlp(args, cache, gh_x, gh_w, gh_b)

    np.testing.assert_allclose(g_x, gh_x, 1e-6, 1e-6)
    for i in range(len(w)):
        np.testing.assert_allclose(g_w[i], gh_w[i], 1e-6, 1e-6)
        np.testing.assert_allclose(g_b[i], gh_b[i], 1e-6, 1e-6)
