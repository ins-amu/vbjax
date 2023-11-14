import os
# for no gpu
# os.environ['CUDA_VISIBLE_DEVICES']=''

import time
import jax, jax.numpy as np
import vbjax as vb

def net(x, p):
    r, v = x
    k, _, mpr_p = p
    c = k*r.sum(), k*v.sum()
    return vb.mpr_dfun(x, c, mpr_p)

def noise(_, p):
    _, sigma, _ = p
    return sigma

_, loop = vb.make_sde(0.01, net, noise)
n_nodes = 164
rv0 = vb.randn(2, n_nodes)
zs = vb.randn(1000, *rv0.shape)

def run(pars, mpr_p=vb.mpr_default_theta):
    k, sig, eta = pars                      # explored pars
    p = k, sig, mpr_p._replace(eta=eta)     # set mpr
    xs = loop(rv0, zs, p)                   # run sim
    std = xs[400:, 0].std()                 # eval metric
    return std

run_batches = jax.jit(jax.vmap(run, in_axes=1))


def bench_cpu():
    run_batches_cores = jax.pmap(jax.vmap(run, in_axes=1), in_axes=1)

    for cores in [8]*10: #[2, 4, 6, 8, 16]:
        for n in [4]: #[2,4,8,16]:
            log_ks, etas = np.mgrid[-9.0:0.0:1j*n, -5.0:-6.0:36j]
            pars = np.c_[np.exp(log_ks.ravel()),np.ones(log_ks.size)*0.2, etas.ravel()].T.copy()
            pars = pars.reshape((3, cores, -1))
            tic = time.time()
            for i in range(50):
                result = run_batches_cores(pars)
            result.block_until_ready()
            toc = time.time()
            iter = 50*log_ks.size*zs.shape[0]
            print(f'{cores} {n} {iter/1e6/(toc-tic):0.2f} Miter/s')
        print()


def bench_gpu():
    for n in [32]*20: #[2,4,8,16,32,48,64]:
        log_ks, etas = np.mgrid[-9.0:0.0:1j*n, -5.0:-6.0:32j]
        pars = np.c_[np.exp(log_ks.ravel()),np.ones(log_ks.size)*0.2, etas.ravel()].T.copy()
        tic = time.time()
        for i in range(50):
            result = run_batches(pars)
        result.block_until_ready()
        toc = time.time()
        iter = 50*log_ks.size*zs.shape[0]
        print(f'{n} {iter/1e6/(toc-tic):0.2f} Miter/s')


if __name__ == '__main__':
    bench_cpu()