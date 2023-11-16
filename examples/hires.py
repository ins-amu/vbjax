import urllib.request

import scipy.io
import jax, jax.numpy as np
import vbjax as vb
import os

mat_fname = 'matrices.mat'
if not os.path.exists(mat_fname):
    urllib.request.urlretrieve(
        'https://github.com/maedoc/popcorn/'
        'raw/main/matrices.mat?download=',
        mat_fname
    )
mat = scipy.io.loadmat('./matrices.mat')
part = 512
SC = vb.csr_to_jax_bcoo(mat['SC'][:part][:,:part].astype('f').tocsr())
LC = vb.csr_to_jax_bcoo(mat['LC'][:part][:,:part].astype('f').tocsr())

# foci = np.argsort((LC.todense()>0).sum(axis=1))[-3:]
foci_points = np.r_[271, 280]#, 760]
foci = [np.r_[_, np.argwhere(LC[_].todense()>0)[:,0]] for _ in foci_points]

def net(xz, p):
    return vb.bvep_dfun(xz, LC@(xz[0] + 1.5)*p[0], p[1])

nv = SC.shape[0]
step, loop = vb.make_ode(0.1, net)
jloop = jax.jit(loop)

xz0 = vb.rand(2, nv)*np.c_[2.5, 2.0].T + np.c_[-2.0, 2.7].T   # full range
xz0 = vb.rand(2, nv)*np.c_[0.3, 0.3].T + np.c_[-1.73, 3.1].T  # around fp
# xz0 = xz0.at[:, foci[0]].set(np.c_[-2.0, 2.0].T)

k = 10.0
theta = vb.bvep_default_theta._replace(x0=-2.2)
x = jloop(xz0, np.r_[:100], (k, theta)) # (T, sv, nn)

do_optimize = False
if do_optimize:
    def loss(xz0hat):
        xhat = loop(xz0hat, np.r_[:200], theta)  # (T, sv, nn)
        sse = np.sum(np.square(x - xhat))
        return sse

    gloss = jax.jit(jax.value_and_grad(loss))

    xz0hat = np.outer(np.r_[-1.73, 3.1], np.ones(nv))
    print('loss on true xz0', gloss(xz0)[0],
          'loss on xz0hat', gloss(xz0hat)[0])
    print('gloss', gloss(xz0hat)[1])

    # optimize it
    from jax.example_libraries.optimizers import adam
    oinit, ostep, oget = adam(1e-3)
    opt = oinit(xz0hat)
    for i in range(0):
        l, g = gloss(oget(opt))
        opt = ostep(i, g, opt)
        if i%100 == 0:
            print(f'iter {i} loss {l:0.2f} ||grad|| {np.linalg.norm(g):0.2f}')

    xz0hat = oget(opt)
    xhat = loop(xz0hat, ts, theta)

import pylab as pl

pl.figure()
for i in range(2):
    pl.subplot(2,1,i+1)
    pl.plot(x[:, i, foci[0]], 'r', alpha=0.5)
    pl.plot(x[:, i, :14], 'k', alpha=0.5)

pl.figure()
pl.plot(x[:, 0, foci[0]], x[:, 1, foci[0]], 'r', alpha=0.5)
pl.plot(x[:, 0, :14], x[:, 1, :14], 'k', alpha=0.5)

pl.figure()
pl.subplot(121)
o = np.argsort(x[:, 0].sum(axis=0))
pl.imshow(x[:, 0, o[-100:]].T, aspect='auto', vmin=-2.0, vmax=1.0)
pl.title('200 most active')
pl.subplot(122)
pl.imshow(x[:, 0, o].T, aspect='auto', vmin=-2.0, vmax=1.0)
pl.title('all')

# pl.figure()
# o = np.argsort(xhat[:, 0].sum(axis=0))[-200:]
# pl.imshow(xhat[:, 0, o].T, aspect='auto')
# pl.title('200 most active, estimated')

pl.show()