import jax
import jax.numpy as np
from jax.example_libraries.optimizers import adam

from .layers import make_dense_layers
from .loops import make_ode


def embed_neural_flow(y, embed_dim=3, latent_dim=[256], iters=8000, step_size=0.01,
                      train_callback=None, oversample=10):
    """
    Embed a dataset using a neural flow.

    Parameters
    ----------
    y : array
        The dataset to embed, with shape (n_time, n_events).
    embed_dim : int
        The dimension of the embedding.
    latent_dim: list of int
        The dimension of the latent neural layers.
    iters : int
        The number of optimization iterations to run.


    Returns
    -------
    z : array
        The embedded dataset.

    """

    # embed y with gradients
    if embed_dim > 0:
        x = [y]
        for i in range( embed_dim - 1 ):
            x.append(np.gradient(x[-1], axis=1))
        x = np.array(x).transpose((2, 0, 1))
    else:
        x = y

    # define neural flow
    nn_p, nn_f = make_dense_layers(x.shape[1], latent_dim, init_scl=1e-3)
    x0 = x[0]
    dt = 1.0 / oversample
    step, loop = make_ode(dt, lambda x, p: nn_f(p, x))
    ts = np.r_[: x.shape[0]*oversample ]
    xp = loop(x0, ts, nn_p)

    # loss function
    def loss(nn_p):
        xp = loop(x0, ts, nn_p)
        return np.sum(np.square(x - xp[::oversample]))
    gloss = jax.jit( jax.grad( loss ) )

    # optimize
    opt = adam(step_size)
    opt_x = opt.init_fn(nn_p)
    trace = []
    trace_grads = []
    best_loss_so_far = np.inf
    best_so_far = nn_p
    for i in range(iters):
        p = opt.params_fn(opt_x)
        g = gloss(p)
        trace.append(loss(p))
        trace_grads.append(g)
        if trace[-1] < best_loss_so_far:
            best_loss_so_far = trace[-1]
            best_so_far = p
        if train_callback is not None:
            xp = loop(x0, ts, p)
            train_callback(i, p, g, trace[-1], x, xp)
        opt_x = opt.update_fn(i, g, opt_x)

    p = best_so_far
    xp = loop(x0, ts, p)

    # return embedded data
    return xp, trace, nn_f, p

