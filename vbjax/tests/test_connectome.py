import jax
import jax.numpy as np

import vbjax as vb


def test_connectome_mvnorm():
    key = jax.random.PRNGKey(42)
    nc, nsc = 5, 100
    SCs = jax.random.normal(key, shape=(nsc, 32, 32))
    results = vb.make_conn_latent_mvnorm(SCs, nc=nc, return_full=True)
    u_mean, u_cov, xfm, u, s, vt, nconf = results
    assert u_mean.size == nc
    assert u_cov.shape == (nc, nc)
    assert u.shape == (nsc, nc)
    assert vt.shape == (nc, SCs.shape[1] ** 2)
    assert nconf < nsc
    *_, nconf2 = vb.make_conn_latent_mvnorm(SCs, nc=nc*2, return_full=True)
    assert nconf2 < nconf
