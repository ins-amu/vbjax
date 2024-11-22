import jax
import jax.numpy as jp
import vbjax as vb
import tqdm


# generate traces from the MPR model w/ different parameter values

def dfun1(x, p):
    c, p = p
    return vb.mpr_dfun(x, (c,0), p)

dt = 0.05
nt = 400
_, loop = vb.make_ode(dt, dfun1)

def run_it(pars):
    c, tau, r0 = pars
    rv0 = jp.r_[r0, -2.0]
    p = c, vb.mpr_default_theta._replace(tau=tau)
    rvs = loop(rv0, jp.r_[:nt], p)
    return rvs

run_them = jax.jit(jax.vmap(run_it))
ng = 16j
cs,taus,r0s=jp.mgrid[0.0:2.0:ng, 1.0:3.0:ng, 0.001:1.0:ng]
pars = jp.c_[cs.ravel(), taus.ravel(), r0s.ravel()]
rvs = run_them(pars)
rvs.block_until_ready()

import pylab as pl
pl.close('all')
pl.semilogy(rvs[:,:,0].T, 'k.-', alpha=0.1)
pl.grid(1)
pl.savefig('scratch.jpg')
# 1/0
# setup a neural ode for this use case
wb, mlp = vb.make_dense_layers(2+2, latent_dims=[128,128], out_dim=2, init_scl=1e-1)
def dfun2(x, wb_pars):
    wb, pars = wb_pars
    x_ = jp.vstack((x, pars[:, :2].T))  # only c & tau
    return mlp(wb, x_)

wb_pars = wb, pars
assert dfun2(rvs[:,0].T, wb_pars).shape == (2, pars.shape[0])

_, mlploop = vb.make_ode(dt, dfun2)
rvs_ = rvs.transpose(1,2,0)

def loss(wb):
    r0 = r0s.ravel()
    x0 = jp.array([r0, -2*jp.ones_like(r0)])
    xs = mlploop(x0, jp.r_[:nt], (wb, pars))
    e = xs - rvs_
    l_x =  jp.mean(jp.square(e))
    l_dx = jp.mean(jp.square(jp.diff(xs, axis=0) - jp.diff(rvs_, axis=0)))
    return l_x

vg = jax.jit(jax.value_and_grad(loss))

print(vg(wb)[0])

# now let's descend the gradient
from jax.example_libraries.optimizers import adam
oinit, oupdate, oget = adam(1e-4)
owb = oinit(wb)
for i in (pbar := tqdm.trange(1000)):
    v, g = vg(oget(owb))
    owb = oupdate(i, g, owb)
    pbar.set_description(f'loss {v:0.5f}')

# since it's trained, we can check how well it works,
test_pars = jp.array([
   [1.93, 2.34, 0.25],
   [0.47, 1.14, 0.29],
 ])
rv_test = run_them(test_pars)  # (len(test_pars), 200, 2)
x0_test = jp.array([test_pars[:,2],
                    jp.ones(len(test_pars))*-2.0])
print(x0_test)
x_test = mlploop(x0_test, jp.r_[:nt], (oget(owb), test_pars))

# show that
pl.figure()
pl.subplot(121)
pl.plot(rv_test[:, :, 0].T, 'k')
pl.plot(x_test[:, 0, :], 'r')
pl.subplot(122)
pl.plot(rv_test[:, :, 1].T, 'k')
pl.plot(x_test[:, 1, :], 'r')
pl.savefig('scratch.jpg')


# two more steps: run a full TVB style simulation


# lastly, run a parameter sweep


