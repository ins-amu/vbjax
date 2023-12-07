#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
**Example 2**: Consider a network of coupled Montbrio model nodes and sweep
over the parameters of the model.

Starting with a few imports

.. literalinclude:: ../examples/01_sweep.py
    :start-after: example-st\u0061rt
    :lines: 1-6
    :caption: 

This example shows how to use the `vbjax` library to simulate a network of
Montbrio model nodes. The network is defined by the function `network` which
takes as arguments the state of the network and the parameters of the model.

.. literalinclude:: ../examples/01_sweep.py
    :start-after: example-st\u0061rt
    :lines: 9-13

The function noise is used to generate the noise term of the stochastic

.. literalinclude:: ../examples/01_sweep.py
    :start-after: example-st\u0061rt
    :lines: 16-18

The function run is used to simulate the network for a given set of parameters.

.. literalinclude:: ../examples/01_sweep.py
    :start-after: example-st\u0061rt
    :lines: 21-26

Then prepare the simulation

.. literalinclude:: ../examples/01_sweep.py
    :start-after: example-st\u0061rt
    :lines: 30

define the number of nodes in the network.

.. literalinclude:: ../examples/01_sweep.py
    :start-after: example-st\u0061rt
    :lines: 31-35

defines the engine to be used for the simulation. If the simulation is run on
the CPU, then the simulation is parallelized over the cores of the CPU using `jax.pmap`.
Otherwise, the simulation is parallelized over the GPU using `jax.vmap`.

then we prepare the network and the noise samples

.. literalinclude:: ../examples/01_sweep.py
    :start-after: example-st\u0061rt
    :lines: 37-39

The rest run the simulation on set of parameters and plot the results.

.. literalinclude:: ../examples/01_sweep.py
    :start-after: example-st\u0061rt
    :lines: 44-75

.. figure:: ../examples/images/sweep.png
    :scale: 100 %


"""
# example-start
import os
import time
import vbjax as vb
import pylab as pl  
import jax, jax.numpy as np
os.makedirs('images', exist_ok=True)


def network(x, p):
    r, v = x
    k, _, mpr_p = p
    c = k*r.sum(), k*v.sum()
    return vb.mpr_dfun(x, c, mpr_p)


def noise(_, p):
    _, sigma, _ = p
    return sigma


def run(pars, mpr_p=vb.mpr_default_theta):
    k, sig, eta = pars                      # explored pars
    p = k, sig, mpr_p._replace(eta=eta)     # set mpr
    xs = loop(rv0, zs, p)                   # run sim
    std = xs[400:, 0].std()                 # eval metric
    return std                              # done


# prepare simulation
n_nodes = 8
using_cpu = jax.local_devices()[0].platform == 'cpu'
if using_cpu:
    run_batches = jax.pmap(jax.vmap(run, in_axes=1), in_axes=0)
else:
    run_batches = jax.vmap(run, in_axes=1)

# prepare network
_, loop = vb.make_sde(0.01, network, noise)
rv0 = vb.randn(2, n_nodes)
zs = vb.randn(1000, *rv0.shape)

tic = time.time()

# sweep sigma but just a few values are enough
sigmas = [0.0, 0.2, 0.3, 0.4]
results = []
ng = vb.cores*4 if using_cpu else 32

for i, sig_i in enumerate(sigmas):
    # create grid of k (on logarithmic scale) and eta
    log_ks, etas = np.mgrid[-9.0:-2.0:1j*ng, -4.0:-6.0:1j*ng]

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


pl.figure(figsize=(8, 2))
pl.figure(figsize=(8,2))
for i, (sig_i, result) in enumerate(zip(sigmas, results)):
    pl.subplot(1, 4, i + 1)
    pl.imshow(result.reshape(log_ks.shape), vmin=0.2, vmax=0.7)
    pl.ylabel('k') if i==0 else (), pl.xlabel('eta')
    pl.title(f'sig = {sig_i:0.1f}')

pl.tight_layout()
pl.savefig('images/sweep.png')

# example-end