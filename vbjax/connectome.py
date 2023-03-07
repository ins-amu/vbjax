"""
Utilities for connectomes.

"""

import jax.numpy as np
import scipy.stats


def make_conn_latent_mvnorm(SCs, nc=10, return_full=False):
    """
    Make a latent multivariate normal distribution over connectomes.
    
    Parameters
    ----------
    SCs : (nconn, n, n) array
        Array of connectomes in a given parcellation.
    nc : int, optional
        Number of components to use for the latent space.
    return_full: bool, optional
        Whether or not to return extra information on the SVD.
    
    Returns
    -------
    u_mean : (nc, ), array_like
        Mean of distribution in latent space.
    u_cov : (nc, nc), array_like
        Covariance of distribution in latent space.
    xfm : function
        Maps latent vector to full connectome.
    u_cov : (nc, nc) array
        Covariance of the multivariate normal. Returned if return_full=True.
    u : (nconn, nconn) array
        Left singular vectors corresponding to connectomes embedded.
        Returned if return_full=True.
    s : (nconn) array
        Singular values. Returned if return_full=True.
    vt : (nconn, n*n) array
        Right singular vectors. Returned if return_full=True.
    nconf : int
        Number of confusions induced by dimensionality reduction.
        Returned if return_full=True.

    """
    nconn, nn, _ = SCs.shape
    u, s, vt = np.linalg.svd(SCs.reshape((nconn, -1)), full_matrices=False)
    u, s, vt = u[:,:nc], s[:nc], vt[:nc]
    rweights = (u @ np.diag(s) @ vt).reshape(SCs.shape)
    sse = np.sum(np.square(rweights[:,None] - SCs).reshape((nconn, nconn, -1)), axis=2)
    # TODO allow nc to be a float specifying percentage of permissible confusions
    nconf = (np.argmin(sse, axis=1) != np.r_[:nconn]).sum()
    u_mean = np.mean(u, axis=0)
    u_cov = np.cov(u.T)
    diag_s = np.diag(s)
    def xfm(new_u):
        return (new_u @ diag_s @ vt).reshape(new_u.shape[:-1] + (nn, nn))
    if return_full:
        return u_mean, u_cov, xfm, u, s, vt, nconf
    return u_mean, u_cov, xfm

