"Jax impl of simulation in fused.ispc.c."

import jax
import jax.numpy as jp
import scipy.sparse
import numpy as np
import gast_model as gm

def make_jp_runsim(
    csr_weights: scipy.sparse.csr_matrix,
    idelays: np.ndarray,
    params: np.ndarray,
    horizon: int,
    rng_seed=43,
    num_item=8,
    num_svar=7,
    num_time=1000,
    dt=0.1,
    num_skip=5,
):
    import jax
    import jax.numpy as jp

    num_out_node, num_node = csr_weights.shape
    horizonm1 = horizon - 1
    j_indices = jp.array(csr_weights.indices)
    j_weights = jp.array(csr_weights.data)
    j_indptr = jp.array(csr_weights.indptr)
    assert idelays.max() < horizon-2
    idelays2 = jp.array(horizon + np.c_[idelays, idelays-1].T)

    _csr_rows = np.concatenate([i*np.ones(n, 'i')
                                for i, n in enumerate(np.diff(csr_weights.indptr))])
    j_csr_rows = jp.array(_csr_rows)
    def cfun(buffer, t):
        wxij = j_weights.reshape(-1,1) * buffer[j_indices, (t - idelays2) & horizonm1]
        cx = jp.zeros((2, num_out_node, num_item))
        cx = cx.at[:, j_csr_rows].add(wxij)
        return cx

    def dfun(x, cx):
        # cx.shape = (3*num_node, num_node, ...)
        Ce_aff, Ci_aff, Cd_aff = cx.reshape(3, num_node, num_item)
        return gm.dopa_dfun(x, (
                            params.we*Ce_aff, 
                            params.wi*Ci_aff, 
                            params.wd*Cd_aff), 
                            params)

    def heun(x, cx, key):
        z = jp.zeros((num_svar, num_node, num_item))
        z = z.at[1:3].set(jax.random.normal(key, (2,num_node,num_item)))
        z = z.at[1].multiply(params.sigma_V)
        z = z.at[2].multiply(params.sigma_u)
        dx1 = dfun(x, cx[0])
        dx2 = dfun(x + dt*dx1 + z, cx[1])
        return x + dt/2*(dx1 + dx2) + z

    def op(sim, T):
        buffer = sim['buf']
        keys = jax.random.split(sim['rng_key'], num_skip+1)
        x = sim['x']
        assert x.shape == (num_svar, num_node, num_item)
        
        for i in range(num_skip):
            t = i + T*num_skip
            cx = cfun(buffer, t)
            x = heun(x, cx, keys[i])
            buffer = buffer.at[:, t % horizon].set(x[0])

        sim['x'] = x
        sim['buf'] = buffer
        sim['rng_key'] = keys[-1]
        return sim, x

    def run_sim_jp(rng_key, init_state):
        buffer = jp.zeros((num_node, horizon, num_item)) + init_state[0]
        init = {
            'buf': buffer,
            'x': jp.zeros((num_svar, num_node, num_item)) + init_state.reshape(-1,1,1),
            'rng_key': rng_key
        }
        ts = jp.r_[:num_time//num_skip]
        
        _, trace = jax.lax.scan(op, init, ts)
        
        return trace

    return jax.jit(run_sim_jp)

