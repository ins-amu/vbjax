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

# setup delays
dt = 0.1
dh = vb.make_delay_helper( weights=jp.log(W+1), lengths=L, dt=dt)

# define parameters for dfun
params = {
    'dh': dh,
    'theta': vb.mpr_default_theta,
    'k': 0.01
}

# define our model
def dfun(buf, rv, t: int, p):
    crv = vb.delay_apply(p['dh'], t, buf)                  # compute delay coupling
    return vb.mpr_dfun(rv, p['k']*crv, p['theta'])        # compute dynamics

# buf should cover all delays + noise for time steps to take
chunk_len = int(10 / dt) # 10 ms
buf = jp.zeros((dh.max_lag + chunk_len, 2, dh.n_from))
buf = buf.at[:dh.max_lag+1].add( jp.r_[0.1,-2.0].reshape(2,1) )

# compile model and enable continuations
_, run_chunk = vb.make_sdde(dt, dh.max_lag, dfun, gfun=1e-3, unroll=10, adhoc=vb.mpr_r_positive)
cont_chunk = vb.make_continuation(run_chunk, chunk_len, dh.max_lag, dh.n_from, n_svar=2, stochastic=True)

# setup time avg and bold monitors
ta_buf, ta_step, ta_samp = vb.make_timeavg((2, dh.n_from))
ta_samp = vb.make_offline(ta_step, ta_samp)
bold_buf, bold_step, bold_samp = vb.make_bold((2, dh.n_from), dt, vb.bold_default_theta)
bold_samp = vb.make_offline(bold_step, bold_samp)

# run chunk w/ monitors
def chunk_ta_bold(sim, key):
    sim['buf'], rv = cont_chunk(sim['buf'], sim['params'], key)
    sim['ta_buf'], ta = ta_samp(sim['ta_buf'], rv)
    sim['bold_buf'], bold = bold_samp(sim['bold_buf'], rv)
    return sim, (ta, bold)

@jax.jit
def run_one_second(sim, key):
    keys = jax.random.split(key, 100) # 100 * 10 ms
    return jax.lax.scan(chunk_ta_bold, sim, keys)

# pack buffers and run it one minute
sim = {
    'params': params,
    'buf': buf,
    'ta_buf': ta_buf,
    'bold_buf': bold_buf
}
ta, bold = [], []
keys = jax.random.split(jax.random.PRNGKey(42), 60)
for i, key in enumerate(tqdm.tqdm(keys)):
    sim, (ta_i, bold_i) = run_one_second(sim, key)
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
