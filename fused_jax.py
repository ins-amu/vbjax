"Jax impl of simulation in fused.ispc.c."

import jax
import jax.numpy as jp
import scipy.sparse
import numpy as np
import gast_model as gm
import vbjax as vb

def make_run_chunk(
    csr_weights: scipy.sparse.csr_matrix,
    idelays: np.ndarray,
    horizon: int,
    chunk_len=5000,
    dt=0.1,
    num_skip=5,
    num_item=8,
    num_svar=7,
    bold_params=None,
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
    
    # BOLD setup
    bold_step, bold_sample = None, None
    if bold_params is not None:
        # BOLD runs at the downsampled rate (dt * num_skip)
        # Convert dt from ms (neural) to s (BOLD)
        bold_init, bold_step, bold_sample = vb.make_bold(
            (num_node, num_item), dt * num_skip * 1e-3, bold_params
        )

    def cfun(buffer, t):
        wxij = j_weights.reshape(-1,1) * buffer[j_indices, (t - idelays2) & horizonm1]
        cx = jp.zeros((2, num_out_node, num_item))
        cx = cx.at[:, j_csr_rows].add(wxij)
        return cx

    def dfun(x, cx, params):
        # cx.shape = (3*num_node, num_node, ...)
        Ce_aff, Ci_aff, Cd_aff = cx.reshape(3, num_node, num_item)
        return gm.dopa_dfun(x, (
                            params.we*Ce_aff, 
                            params.wi*Ci_aff, 
                            params.wd*Cd_aff), 
                            params)

    def heun(x, cx, key, params):
        z = jp.zeros((num_svar, num_node, num_item))
        z = z.at[1:3].set(jax.random.normal(key, (2,num_node,num_item)))
        z = z.at[1].multiply(params.sigma_V)
        z = z.at[2].multiply(params.sigma_u)
        dx1 = dfun(x, cx[0], params)
        dx2 = dfun(x + dt*dx1 + z, cx[1], params)
        return x + dt/2*(dx1 + dx2) + z

    def op(sim, _):
        buffer = sim['buf']
        t = sim['t']
        keys = jax.random.split(sim['rng_key'], num_skip+1)
        x = sim['x']
        params = sim['params']
        
        r_accum = jp.zeros_like(x[0])

        # Re-defining micro_op to take i for key access
        def micro_op_idx(carry, i):
            buffer, x, t, r_accum = carry
            cx = cfun(buffer, t)
            x = heun(x, cx, keys[i], params)
            buffer = buffer.at[:, t & horizonm1].set(x[0])
            r_accum += x[0]
            t += 1
            return (buffer, x, t, r_accum), None

        (buffer, x, t, r_accum), _ = jax.lax.scan(micro_op_idx, (buffer, x, t, r_accum), jp.arange(num_skip))

        sim['x'] = x
        sim['buf'] = buffer
        sim['t'] = t
        sim['rng_key'] = keys[-1]
        
        if bold_params is not None:
            r_avg = r_accum / num_skip
            sim['bold'] = bold_step(sim['bold'], r_avg)
            
        return sim, None

    scan_len = chunk_len // num_skip

    def run_chunk(state, key, params):
        buffer, x, t = state[:3]
        bold_state = state[3] if bold_params is not None else None
        
        sim_init = {
            'buf': buffer,
            'x': x,
            't': t,
            'rng_key': key,
            'params': params,
        }
        if bold_params is not None:
            sim_init['bold'] = bold_state

        ts = jp.arange(scan_len)
        
        final_sim, _ = jax.lax.scan(op, sim_init, ts)
        
        # No Buffer Roll needed.
        new_state = (final_sim['buf'], final_sim['x'], final_sim['t'])
        
        bold_val = None
        if bold_params is not None:
            new_bold_state, bold_val = bold_sample(final_sim['bold'])
            new_state = new_state + (new_bold_state,)
        else:
            new_state = new_state + (None,)

        return new_state, bold_val

    return run_chunk

