import numpy as np
import numpyro
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
import pytest

import vbjax


def setup_model():
    # setup model
    lmax, nlat, nlon = 31, 32, 64
    lc = vbjax.make_shtdiff(lmax=lmax, nlat=nlat, nlon=nlon)
    x0 = jax.random.normal(jax.random.PRNGKey(42), (nlat, nlon))
    x1 = lc(x0)
    ts = jnp.r_[:100]*0.01
    _, run = vbjax.make_ode(0.01, lambda x, k: -x + k*lc(x))
    k = 1.0
    xt = run(x0, ts, k)

    # parameter to recover
    x0 = x0.at[:15,:15].set(3.0)
    xt = run(x0, ts, k)

    # bayesian model
    import numpyro.distributions as dist
    def logp(xt=None):
        x0h = numpyro.sample('x0h', dist.Normal(jnp.zeros((nlat, nlon)), 1))
        xth_mu = run(x0h, ts, k)
        numpyro.sample('xth', dist.Normal(xth_mu, 1), obs=xt)
    return x0, xt, logp


@pytest.mark.slow
def test_sht_ode_nuts():
    x0, xt, logp = setup_model()
    nuts_kernel = NUTS(logp)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, xt=xt)
    x0h = mcmc.get_samples()['x0h']

    # check effective sample size
    ess = numpyro.diagnostics.effective_sample_size(x0h.reshape((1, 500, -1)))
    assert ess.min() > 100

    # check sbc measures
    shrinkage, zscore = vbjax.shrinkage_zscore(x0, x0h, 1)
    assert shrinkage.min() > 0.7
    assert zscore.max() < 1.5
