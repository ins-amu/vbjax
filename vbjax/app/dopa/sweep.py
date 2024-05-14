# functions for running sweeps over the dopa model and collecting results

import jax
import jax.numpy as jp
import vbjax as vb


# TODO can't continue simulations b/c init doesn't vmap w/ params
# TODO need to run long time tavg and bold
# TODO allow predefined set of params for e.g. random sample
# TODO regional parameters?
# TODO abstract over simulation configuration e.g. noise
# TODO ISPC impl for benchmark comparison
# TODO use delays if available
# TODO implement memory limits & test them
# TODO memmap raw/tavg/bold direct to file
# TODO automate inner vs outer loop over parameters



def sweep_node(init, params, T=10.0, dt=0.001,
               sigma=1e-3, seed=42, cores=4,
               tavg=None):
    "Run sweep for single dopa node on params matrix"

    # setup grid for parameters
    pkeys, pgrid = vb.tuple_meshgrid(params)
    pshape, pravel = vb.tuple_ravel(pgrid)

    # distribute params for cpu; doesn't work for now
    if vb.is_cpu:
        pravel = vb.tuple_shard(pravel, cores)

    # setup model
    f = lambda x, p: vb.dopa_dfun(x, (0,0,0), p)
    _, loop = vb.make_sde(dt, f, sigma, adhoc=vb.dopa_r_positive)

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


def sweep_network(init, params, Ci, Ce, Cd,
                  T=10.0, dt=0.01, sigma=1e-3, seed=42, cores=4):
    "Run sweep for single dopa node on params matrix"

    # check & convert connectivities
    assert Ci.shape == Ce.shape == Cd.shape
    n_nodes = Ci.shape[0]
    Ci, Ce, Cd = [jp.array(_.astype('f')) for _ in (Ci, Ce, Cd)]

    # expand initial conditions if required
    if init.ndim == 1:
        init = jp.outer(init, jp.ones(n_nodes))

    # setup grid for parameters
    pkeys, pgrid = vb.tuple_meshgrid(params)
    pshape, pravel = vb.tuple_ravel(pgrid)

    # distribute params for cpu; doesn't work for now
    if vb.is_cpu:
        pravel = vb.tuple_shard(pravel, cores)

    # setup model
    _, loop = vb.make_sde(dt, vb.dopa_net_dfun, sigma, adhoc=vb.dopa_r_positive)

    # assume same inits and noise for all params
    key = jax.random.PRNGKey(seed)
    nt = int(T / dt)
    dw = jax.random.normal(key, (nt, 6, n_nodes))
    
    # run sweep
    runv = jax.vmap(lambda p: loop(init, dw, (Ci,Ce,Cd,p)))
    run_params = jax.jit(jax.vmap(runv) if vb.is_cpu else runv)
    ys = run_params(pravel)

    # reshape the resulting time series
    # assert ys.shape == (pravel[0].size, nt, 6)
    ys = ys.reshape(pshape + (nt, 6, n_nodes))

    return pkeys, ys


if __name__ == '__main__':

    # TODO set up an argparser to have a cli e.g. on slurm
    pass
