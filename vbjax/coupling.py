"""
Predefined couplings of interest for most applications.

"""

from collections import namedtuple
import jax.numpy as jp


DelayHelper = namedtuple('DelayHelper', 'Wt lags ix_lag_from max_lag n_to n_from')

def make_delay_helper(weights, lengths, dt=0.1, v_c=10.0) -> DelayHelper:
    """Construct a helper with auxiliary variables for applying 
    delays to a buffer.
    """
    n_to, n_from = weights.shape
    lags = jp.floor(lengths / v_c / dt).astype('i')
    ix_lag_from = jp.tile(jp.r_[:n_from], (n_to, 1))
    max_lag = lags.max() + 1
    Wt = weights.T[:,:,None] # enable bcast for coupling vars
    dh = DelayHelper(Wt, lags, ix_lag_from, max_lag, n_to, n_from)
    return dh

def delay_apply(dh: DelayHelper, t, buf):
    """Apply delays to buffer `buf` at time `t`.
    """
    return (dh.Wt * buf[t - dh.lags, :, dh.ix_lag_from]).sum(axis=1).T

# TODO impl sparse delay_apply
