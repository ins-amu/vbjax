"""
Predefined couplings of interest for most applications.

"""

import jax.numpy as np


def make_delayed_coupling(weights, delay_steps, pre, post, nh, isvar):
    """
    Construct a dense delayed coupling function. 

    Parameters
    ==========
    weights : array
        Coupling weights.
    delay_steps : array
        Number of delay steps per connection i, j.

    ...
    To be finished
    ...

    Notes
    =====

    - This construction assumes a particular layout for the history
      buffer: xt.shape == (nh+1+nt, nsvar, nnode, ...). 

    """
    nn = weights.shape[0]
    nodes = np.tile(np.r_[:nn], (nn, 1))
    def cfun(t, xt, x, params):
        dx = xt[nh + t - delay_steps, isvar, nodes]
        xij = pre(dx, x, params)
        gx = (weights * xij).sum(axis=1)
        return post(gx, params)
    return cfun


def make_linear_cfun(SC, a=1.0, b=0.0):
    """Construct a linear coupling function with slope `a` and offset `b`.
    """
    def cfun(xj):
        "Compute linear coupling for pre-synaptic state variables xj."
        if xj.ndim == 1:  # no delays
            gx = SC @ xj
        elif xj.ndim == 2:  # delays
            gx = np.sum(SC * xj, axis=1)
        return a*gx + b
    return cfun


def make_diff_cfun(SC, a=1.0, b=0.0):
    """Construct a linear difference coupling."""
    nn = np.r_[:SC.shape[0]]
    diffdiag = np.diag(SC) - SC.sum(axis=1)
    SC_ = SC.at[nn,nn].set(diffdiag)
    # fix diagonal according to trick
    return make_linear_cfun(SC_, a=a, b=b)
