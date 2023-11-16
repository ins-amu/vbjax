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
n_nodes = 8
rv0 = vb.randn(2, n_nodes)
zs = vb.randn(1000, *rv0.shape)

def run(pars, mpr_p=vb.mpr_default_theta):
    k, sig, eta = pars                      # explored pars
    p = k, sig, mpr_p._replace(eta=eta)     # set mpr
    xs = loop(rv0, zs, p)                   # run sim
    std = xs[400:, 0].std()                 # eval metric
    return std                              # done


using_cpu = jax.local_devices()[0].platform == 'cpu'
if using_cpu:
    run_batches = jax.pmap(jax.vmap(run, in_axes=1), in_axes=0)
else:
    run_batches = jax.vmap(run, in_axes=1)

import time

tic = time.time()

# sweep sigma but just a few values are enough
sigmas = [0.0, 0.2, 0.3, 0.4]
results = []
for i, sig_i in enumerate(sigmas):
    # create grid of k (on logarithmic scale) and eta
    log_ks, etas = np.mgrid[-9.0:-2.0:64j, -4.0:-6.0:64j]

    # reshape grid to big batch of values
    pars = np.c_[
        np.exp(log_ks.ravel()),
        np.ones(log_ks.size)*sig_i,
        etas.ravel()].T.copy()

    # cpu w/ pmap expects a chunk for each core
    if using_cpu:
        pars = pars.reshape((3, vb.cores, -1)).transpose((1, 0, 2))

    # now run
    result = run_batches(pars).block_until_ready()
    results.append(result)
toc = time.time()
print(f'elapsed time for sweep {toc - tic:0.1f} s')


import pylab as pl
pl.figure(figsize=(8,2))
for i, (sig_i, result) in enumerate(zip(sigmas, results)):
    pl.subplot(1, 4, i + 1)
    pl.imshow(result.reshape(log_ks.shape), vmin=0.2, vmax=0.7)
    pl.ylabel('k') if i==0 else (), pl.xlabel('eta')
    pl.title(f'sig = {sig_i:0.1f}')
pl.show()