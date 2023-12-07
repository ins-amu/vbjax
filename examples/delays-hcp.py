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

def rgt0(rv, p):
    r, v = rv
    return jp.array([ r*(r>0), v ])

# compile dfun w/ heun sdde for running a chunk
_, run_chunk = vb.make_sdde(dt, max_lag, dfun, gfun=1e-3, unroll=10, adhoc=rgt0)

# we'll run chunk of time this long
chunk_len = int(10 / dt) # 10 ms

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

# jit the buffer updates
# this kinda sucks but it's not model specific, so perfectly reusable
def make_continue_chunk(run_chunk, chunk_len, max_lag, n_from, n_svar):
    # need to be compile time constants for dynamic_*
    i0 = chunk_len - 1
    i1 = max_lag + 1
    @jax.jit
    def continue_chunk(buf, p, key):
        get, set = jax.lax.dynamic_slice, jax.lax.dynamic_update_slice
        # buf = buf.at[:max_lag+1].set( buf[-(max_lag+1):] )
        buf = set(buf, get(buf, (i0,0,0), (i1,n_svar,n_from)), (0,0,0))
        # buf = buf.at[max_lag+1:].set( vb.randn(chunk_len-1, 2, n_from, key=key) )
        buf = set(buf, vb.randn(chunk_len-1, n_svar, n_from, key=key), (i1,0,0))
        buf, rv = run_chunk(buf, p)
        return buf, rv
    return continue_chunk
continue_chunk = make_continue_chunk(run_chunk, chunk_len, max_lag, n_from, n_svar=buf.shape[1])

# setup time avg and bold monitors
ta_buf, ta_step, ta_samp = vb.make_timeavg((2, n_from))
ta_samp = vb.make_offline(ta_step, ta_samp)
bold_buf, bold_step, bold_samp = vb.make_bold((2, n_from), dt, vb.bold_default_theta)
bold_samp = vb.make_offline(bold_step, bold_samp)

# now scan over rng keys (?)
def chunk_ta_bold(bufs, key):
    buf, ta_buf, bold_buf = bufs
    buf, rv = continue_chunk(buf, p, key)
    ta_buf, ta = ta_samp(ta_buf, rv)
    bold_buf, bold = bold_samp(bold_buf, rv)
    return (buf, ta_buf, bold_buf), (ta, bold)

@jax.jit
def run_one_second(bufs, key):
    keys = jax.random.split(key, 100) # 100 * 10 ms
    return jax.lax.scan(chunk_ta_bold, bufs, keys)

# pack buffers and run it one minute
bufs = buf, ta_buf, bold_buf
ta, bold = [], []
keys = jax.random.split(jax.random.PRNGKey(42), 60)
for i, key in enumerate(tqdm.tqdm(keys)):
    bufs, (ta_i, bold_i) = run_one_second(bufs, key)
    ta.append(ta_i)
    bold.append(bold_i)
ta = jp.array(ta).reshape((-1, 2, 70))
bold = jp.array(bold).reshape((-1, 2, 70))

# look at it
import matplotlib.pyplot as pl
pl.subplot(221); pl.plot(ta[:, 0], 'k', alpha=0.2); pl.title('r t avg'); pl.grid(1)
pl.subplot(222); pl.plot(bold[:, 0], 'b', alpha=0.2); pl.title('r bold'); pl.grid(1)
pl.subplot(223); pl.plot(ta[:, 1], 'k', alpha=0.2); pl.title('r t avg'); pl.grid(1)
pl.subplot(224); pl.plot(bold[:, 1], 'b', alpha=0.2); pl.title('r bold'); pl.grid(1)
pl.show()
