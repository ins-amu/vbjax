import jax
import jax.numpy as jp
import vbjax as vb
import tqdm


# generate traces from the MPR model w/ different parameter values

def dfun1(x, p):
    c, p = p
    return vb.mpr_dfun(x, (c,0), p)

dt = 0.1
_, loop = vb.make_ode(dt, dfun1)

def run_it(pars):
    c, tau, r0 = pars
    rv0 = jp.r_[r0, -2.0]
    p = c, vb.mpr_default_theta._replace(tau=tau)
    nt = 200
    rvs = loop(rv0, jp.r_[:nt], p)
    return rvs

run_them = jax.jit(jax.vmap(run_it))
ng = 4j
cs,taus,r0s=jp.mgrid[0.0:2.0:ng, 1.0:3.0:ng, 0.001:1.0:ng]
pars = jp.c_[cs.ravel(), taus.ravel(), r0s.ravel()]
rvs = run_them(pars)
rvs.block_until_ready()

import pylab as pl
pl.close('all')
pl.semilogy(rvs[:,:,0].T, 'k.-', alpha=0.1)
pl.grid(1)
pl.savefig('scratch.jpg')

# setup a neural ode for this use case
wb, mlp = vb.make_dense_layers(2+2, latent_dims=[16,16], out_dim=2)
def dfun2(x, wb_pars):
    wb, pars = wb_pars
    x_ = jp.vstack((x, pars[:, :2].T))  # only c & tau
    return mlp(wb, x_)

wb_pars = wb, pars
assert dfun2(rvs[:,0].T, wb_pars).shape == (2, pars.shape[0])

_, mlploop = vb.make_ode(dt, dfun2)
rvs_ = rvs.transpose(1,2,0)
def loss(wb):
    # init with r0s
    x0 = rvs[:,0].T.at[0].set(pars[:,2])
    xs = mlploop(x0, jp.r_[:200], (wb, pars))
    e = xs - rvs_
    l_x =  jp.mean(jp.square(e))
    l_dx = jp.mean(jp.square(jp.diff(e, axis=0)))
    return l_x# + l_dx

vg = jax.jit(jax.value_and_grad(loss))

print(vg(wb)[0])

# now let's descend the gradient
from jax.example_libraries.optimizers import adam
oinit, oupdate, oget = adam(1e-2)
owb = oinit(wb)
for i in (pbar := tqdm.trange(300)):
    v, g = vg(oget(owb))
    owb = oupdate(i, g, owb)
    pbar.set_description(f'loss {v:0.2f}')

# since it's trained, we can check how well it works,
test_pars = jp.array([
   [1.93, 2.34, 0.25],
   [0.47, 1.14, 0.29],
 ])
rv_test = run_them(test_pars)  # (len(test_pars), 200, 2)
x0_test = jp.array([test_pars[:,2],
                    jp.ones(len(test_pars))*-2.0])
x_test = mlploop(x0_test, jp.r_[:200], (oget(owb), test_pars))

# show that
pl.figure()
pl.plot(rv_test[:, :, 0].T, 'k')
pl.plot(x_test[:, 0, :], 'r')
pl.savefig('scratch.jpg')
