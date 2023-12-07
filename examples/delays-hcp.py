# example showing time delays w/ hcp connectomes from scratch

import os
import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb
import tqdm

npz_fname = 'hcp-dk-dest-jub.npz'
if not os.path.exists(npz_fname):
    def load(fname):
        try:
            return np.loadtxt(fname)
        except ValueError:
            return np.loadtxt(fname, delimiter=',')
    # download and unzip from
    # https://search.kg.ebrains.eu/instances/f16e449d-86e1-408b-9487-aa9d72e39901
    parcs = '070-DesikanKilliany,150-Destrieux,294-Julich-Brain'.split(',')
    npz = {}
    for parc in parcs:
        path = f'../popovych-hcp-connectomes/{parc}/1StructuralConnectivity'
        counts = np.array([load(f'{path}/{i:03d}/Counts.csv') for i in range(200)])
        lengths = np.array([load(f'{path}/{i:03d}/Lengths.csv') for i in range(200)])
        npz[f'{parc}'] = np.array([counts, lengths]).astype('f')
    np.savez(npz_fname, **npz)
npz = np.load(npz_fname)

# load one connectome
W, L = jp.array(npz['070-DesikanKilliany'][:,0])
n_to, n_from = W.shape

# setup aux vars for delays
dt = 0.1
v_c = 10.0 # m/s
lags = jp.floor(L / v_c / dt).astype('i')
ix_lag_from = jp.tile(jp.r_[:n_from], (n_to, 1))
max_lag = lags.max() + 1
Wt = jp.log(1+W.T[:,:,None]) # enable bcast for crv


def dfun(buf, rv, t: int, p):
    # we could close over the vars or pass in like so:
    Wt, lags, ix_lag_from, mpr_theta, k = p
    crv = (Wt * buf[t - lags, :, ix_lag_from]).sum(axis=1).T
    return vb.mpr_dfun(rv, k*crv, mpr_theta)

# compile dfun w/ heun sdde for running a chunk
_, run_chunk = vb.make_sdde(dt, max_lag, dfun, gfun=1e-3, unroll=4)

# we'll run chunk of time this long
chunk_len = 1000

# buf should cover all delays + noise for time steps
buf = jp.zeros((max_lag + chunk_len, 2, n_from))

# set initial history
buf = buf.at[:max_lag+1].add( jp.r_[0.1,-2.0].reshape(2,1) )

# set noise samples
buf = buf.at[max_lag+1:].set( vb.randn(chunk_len-1, 2, n_from) )

# pack parameters (could/should be dict / dataclass)
k = 0.01
p = Wt, lags, ix_lag_from, vb.mpr_default_theta, k

# run it
buf, rv = run_chunk(buf, p)

# check
assert buf.shape[0] == (max_lag + chunk_len)
assert rv.shape == (chunk_len, 2, n_from)

# in a loop, careful with rng keys
rng_keys = jax.random.split(jax.random.PRNGKey(42), 100)
rs = [rv[::10,0]]

def make_continue_chunk(chunk_len, max_lag, n_from):
    # need to be compile time constants for dynamic_*
    i0 = chunk_len - 1
    i1 = max_lag + 1

    @jax.jit
    def continue_chunk(buf, p, key):
        get, set = jax.lax.dynamic_slice, jax.lax.dynamic_update_slice

        # buf = buf.at[:max_lag+1].set( buf[-(max_lag+1):] )
        buf = set(buf, get(buf, (i0,0,0), (i1,2,n_from)), (0,0,0))

        # buf = buf.at[max_lag+1:].set( vb.randn(chunk_len-1, 2, n_from, key=key) )
        buf = set(buf, vb.randn(chunk_len-1, 2, n_from, key=key), (i1,0,0))

        return run_chunk(buf, p)
    return continue_chunk
continue_chunk = make_continue_chunk(chunk_len, max_lag, n_from)

# todo convert to scan
for key in tqdm.tqdm(rng_keys):
    # outside jax.jit these copy, but jit makes them inplace ops
    buf, rv = continue_chunk(buf, p, key)
    rs.append(rv[::10,0])
rs = jp.array(rs) # would happen automagically if loop done w/ scan
print(rs.shape)
rs = rs.reshape(-1, n_from) # reshape still required

# look at it
import matplotlib.pyplot as pl
pl.plot(rs, 'k', alpha=0.2)
pl.show()
