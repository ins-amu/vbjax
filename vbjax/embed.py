import itertools
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


def embed_polynomial(x, y=None, max_order=2):
    """
    Embed a dataset using a polynomial basis.

    Parameters
    ----------
    x : array
        The dataset to embed, with shape (n_dim, n_sample).
    y : array, optional
        The target data for a polynomial regression, (n_target, n_sample).
    max_order : int
        The maximum order of the polynomial basis.

    Returns
    -------
    basis : array
        The polynomial basis, with shape (n_basis, n_sample).
    coef : array, optional
        The coefficients of the polynomial regression, with shape (n_basis, n_target).
        This is only returned if `y` is provided.

    Notes
    -----
    The loss can be computed by 
    >>> np.square(coef.T @ basis - Y).sum()

    """

    n_dim = len(x)
    n_samp = len(x.T)
    basis = [np.ones((1, n_samp))]

    # construct columns of A matrix
    for order in range(1, max_order+1):
        all_idx = itertools.product(*([range(n_dim)] * order))
        x_idx = list({tuple(sorted(idx)): None for idx in all_idx}.keys())
        basis.append(np.prod(x[np.array(x_idx)], axis=1))
    basis = np.concatenate(basis, axis=0)

    if y is None:
        return basis

    # compute least square solution
    coef, *_ = np.linalg.lstsq(basis.T, y.T, rcond=None)

    return basis, coef


def embed_gradient(x, n_grads=2, axis=1):
    """
    Embed a dataset using repeated `np.gradient` calls to compute
    derivatives, usually for time series data.

    Parameters
    ----------
    x : array
        The dataset to embed, shape (n_dim, n_time, ...).
    n_grads : int
        The number of derivatives to compute.

    Returns
    -------
    basis : array
        The derivative basis, with shape (n_grads, n_dim, n_time, ...).

    """
    xs = [x]
    for i in range(n_grads):
        xs.append(np.gradient(xs[-1], axis=1))
    return np.array(xs)


def embed_autoregress(x, n_lag=3, lag=1):
    """
    Embed a dataset as an autoregressive process,
    usually for time series data.

    Parameters
    ----------
    x : array
        The dataset to embed, shape (n_dim, n_time, ...).
    n_lag : int
        The number of lags to use.
    lag : int, default 1
        The lag to use.

    Returns
    -------
    basis : array
        The autoregressive basis, with shape (n_lag, n_dim, n_time - lag*n_lag, ...)

    """
    xs = np.array([x[:, i*lag:(-n_lag+i)*lag] for i in range(n_lag)])
    return xs
