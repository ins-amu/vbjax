import numpy as np
import ctypes as ct
import scipy.sparse
import time

if __name__ == '__main__':


    from fused_np import run_sim_np
    from fused_jax import make_jp_runsim
    from simdata import make_sim
    # from compiled import load_c, load_ispc

    np.random.seed(42)

    num_item = 16384*2
    num_node = 90
    num_skip = 20
    dt = 0.1
    sparsity = 0.3
    horizon = 256
    num_time = int(1e3/dt)
    horizonm1 = horizon - 1
    sim_params = np.zeros((3, num_item), 'f')
    sim_params[0] = 1.001
    sim_params[1] = 1.0
    sim_params[2] = np.logspace(-1.8, -2.0, num_item)/num_node*80 # k
    z_scale = np.sqrt(dt)*np.r_[0.01, 0.1].astype('f')*1e-8

    weights, lengths = np.random.rand(2, num_node, num_node).astype('f')
    lengths[:] *= 0.8
    lengths *= (horizon*dt*0.8)
    zero_mask = weights < (1-sparsity)
    weights[zero_mask] = 0
    csr_weights = scipy.sparse.csr_matrix(weights)
    idelays = (lengths[~zero_mask]/dt).astype('i')+2

    run_args = csr_weights, idelays, sim_params, z_scale, horizon
    run_kwargs = dict(num_item=num_item, num_node=num_node, num_time=num_time,
                      dt=dt, num_skip=num_skip)

    # lib_c = load_c()
    # lib_ispc = load_ispc(node_wise=False, isa='avx2-i32x8')

    traces = {}

    tic = time.time()
    # traces['numpy'] = run_sim_np(*run_args, **run_kwargs)[:-1]
    tok = time.time()
    print(tok - tic, 's', num_time*num_item/(tok-tic), 'iter/s numpy')
    # ntnp = traces['numpy'].shape[0]

    run_sim_jp = make_jp_runsim(*run_args, **run_kwargs)
    import jax
    j_key = jax.random.PRNGKey(42)
    run_sim_jp(j_key).block_until_ready()
    time.sleep(5)
    tic = time.time()
    traces['jax'] = run_sim_jp(j_key).block_until_ready()
    tok = time.time()
    print(tok - tic, 's', num_time*num_item/(tok-tic), 'iter/s jax')

    impls = [
        # ('c', lib_c),
        #('ispc', lib_ispc),
    ]

    for i, (key, lib) in enumerate(impls):
        sim, sim_arrays = make_sim(*run_args, **run_kwargs)
        traces[key] = sim_arrays[0]
        tic = time.time()
        sim_ref = ct.byref(sim)
        lib.run_batches(sim_ref)
        tok = time.time()
        print(tok - tic, 's', num_time*num_item/(tok-tic), f'iter/s {key}')

    # import matplotlib.pyplot as pl
    # nimpl = len(traces)
    # pl.figure(figsize=(7, 8))
    # for i_col, key in enumerate(traces.keys()):
    #     for i_row in range(num_item):
    #         pl.subplot(num_item, nimpl, i_col+1 + nimpl*i_row)
    #         pl.plot(traces[key][:, 0, :, i_row], 'k', alpha=0.4)
    #         pl.plot(traces[key][:, 1, :, i_row], 'b', alpha=0.4)
    #         pl.grid(1)
    # pl.tight_layout()
    # pl.savefig('fused.jpg', dpi=300)

    # import matplotlib.pyplot as pl
    # pl.figure(figsize=(6,10))
    # pl.subplot(211)
    # for i, (k, v) in enumerate(traces.items()):
    #     print(k)
    #     pl.plot(v[:ntnp, 0, :5, 0]+np.r_[:5], 'rgb'[i]+'.-', alpha=0.3)
    # pl.subplot(212)
    # pl.plot(traces['c'][:ntnp, 0, :5, 0] - traces['numpy'][:ntnp, 0, :5, 0]+np.r_[:5]*1e-2, 'k')
    # [pl.axvline(_/num_skip, alpha=0.3) for _ in idelays]
    # pl.grid()
    # pl.tight_layout()
    # pl.savefig('fused.jpg', dpi=600)


    # np.testing.assert_allclose(traces['c'][:ntnp], traces['numpy'], 1e-6, 1e-6)
    # np.testing.assert_allclose(traces['ispc'][1:], traces['numpy'], 1e-6, 1e-6)

