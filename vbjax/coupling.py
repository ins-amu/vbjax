"""
Predefined couplings of interest for most applications.
"""

import jax.numpy as np


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