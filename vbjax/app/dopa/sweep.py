# functions for running sweeps over the dopa model and collecting results

import jax
import jax.numpy as jp
import vbjax as vb


def sweep_node(init, params, T=10.0, dt=0.01, sigma=1e-3, seed=42, cores=4):
    "Run sweep for single dopa node on params matrix"

    # setup grid for parameters
    pkeys, pgrid = vb.tuple_meshgrid(params)
    pshape, pravel = vb.tuple_ravel(pgrid)

    # distribute params for cpu; doesn't work for now
    if vb.is_cpu:
        pravel = vb.tuple_shard(pravel, cores)

    # setup model
    f = lambda x, p: vb.dopa_dfun(x, (0,0,0), p)
    _, loop = vb.make_sde(dt, f, sigma)

    # assume same inits and noise for all params
    key = jax.random.PRNGKey(seed)
    nt = int(T / dt)
    dw = jax.random.normal(key, (nt, 6))
    
    # run sweep
    runv = jax.vmap(lambda p: loop(init, dw, p))
    run_params = jax.jit(jax.vmap(runv) if vb.is_cpu else runv)
    ys = run_params(pravel)

    # reshape the resulting time series
    # assert ys.shape == (pravel[0].size, nt, 6)
    ys = ys.reshape(pshape + (nt, 6))

    return pkeys, ys


if __name__ == '__main__':

    # start with default parameters        
    params = vb.dopa_default_theta

    # sweeps over Km and Vmax
    params = params._replace(
        Km=jp.r_[100:200:32j],
        Vmax=jp.r_[1000:2000:32j],
        k=jp.r_[0.1:0.2:16j],
        beta=jp.r_[0.1:0.7:16j],
        )

    # initial conditions
    y0 = jp.array([0., -2.0, 0.0, 0.0, 0.0, 0.0])

    # run sweep
    end_time = 256.0
    pkeys, ys = sweep_node(y0, params, T=end_time, cores=4)

    print(pkeys, ys.shape)