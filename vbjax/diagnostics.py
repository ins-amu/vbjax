import jax.numpy as np


def shrinkage_zscore(x, x_hat, x_prior):
    """
    Computes the posterior shrinkage and z-score for 
    a parameter x.

    Parameters
    ----------
    x : array
        The true parameter.
    x_hat : array
        The posterior samples of the parameter.
    x_prior : array
        The prior standard deviation of the parameter.

    Returns
    -------
    shrinkage : array
        The posterior shrinkage.
    zscore : array
        The posterior z-score.

    """
    # compute posterior mean
    x_mean = x_hat.mean(0)

    # compute posterior standard deviation
    x_std = x_hat.std(0)

    # compute posterior shrinkage
    shrinkage = 1 - x_std / x_prior

    # compute posterior z-score
    zscore = (x_mean - x) / x_std

    return shrinkage, zscore
