import os
import sys
import tqdm
import numpy as np
import jax
import jax.numpy as jp
import scipy.sparse
import pylab as pl

import vbjax as vb
import gast_model as gm
import fused_jax

# load connectivity and lengths
W70 = np.loadtxt(f'Counts.csv')
nn, _ = W70.shape
np.fill_diagonal(W70, 0)
L70 = np.loadtxt(f'Lengths.csv')

# create random masks for now
Mi = np.random.rand(nn, nn) < 0.01
Md = np.random.rand(nn, nn) < 0.01
Mi.sum(), Md.sum()

# create scaled and masked connectivities
Ce70 = np.log(W70+1)/np.log(W70+1).max()
Ci70 = Ce70 * Mi
Cd70 = Ce70 * Md
print(f'W70: {W70.max(), W70.mean()}')
print(f'Ce70: {Ce70.max(), Ce70.mean()}')

# prep sparse matrices. for 3 connectivities which have the same
# cvar, it's more efficient to stack them so we do a single retrieval
# from the delay buffer.  then we sparsify them.
Ceid70 = np.vstack([Ce70, Ci70, Cd70])
Leid70 = np.vstack([L70, L70, L70])
Seid70 = scipy.sparse.csr_matrix(Ceid70)
print(f'Seid70.shape {Seid70.shape}, Seid70.indices.shape {Seid70.indices.shape}')

# compute discrete delays
v_c = 10.0
dt = 0.1
idelays = (Leid70[Ceid70 != 0.0] / v_c / dt).astype(np.uint32)
idelays.max(), idelays.shape

# set up simulation parameters
z_scale = jp.array([0., 0.1, 0.01, 0., 0., 0., 0.]).reshape(-1,1)
n_svar = 7
init_state = jp.array([0.01, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(n_svar, 1)
num_time = 10000
num_skip = 10
num_item = 32
Jsa = jp.r_[10:100:1j*num_item]
sigma_V = 10**jp.r_[-1:0:1j*num_item]

node_theta = gm.dopa_default_theta._replace(
    I=0, Ja=0, Jsa=Jsa, Jsg=0, Jdopa=0, Jg=0, Rd=0,
    sigma_V=sigma_V
)

run_sim_jp = fused_jax.make_jp_runsim(
    Seid70,
    idelays,
    node_theta,
    horizon=256,
    num_item=num_item,
    dt=dt,
    num_skip=num_skip,
    num_time=num_time,
)

rng_key = jax.random.PRNGKey(42)
xs = run_sim_jp(rng_key, init_state)
print(f'xs.shape {xs.shape}')

pl.imshow(xs[:,0].mean(axis=0), aspect='auto'); pl.colorbar()
pl.show()
