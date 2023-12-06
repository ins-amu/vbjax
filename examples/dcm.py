import pylab as pl
import vbjax as vb
import jax
import jax.numpy as jp
import tqdm

n = 5

# assume damped dynamics at each node
A = -jp.identity(n)

# two conditions
B = jp.zeros((n, n, 2))
C = jp.zeros((n, B.shape[2]))

# first condition induce oscillation between nodes 1 & 2
B = B.at[0,1,0].set(1)
B = B.at[1,0,0].set(-1)

# second condition induce faster oscillation between nodes 3 & 4
B = B.at[3,2,1].set(3)
B = B.at[2,3,1].set(-3)

# second condition shifts node 5 fixed point up by 0.5
C = C.at[4, 1].set(0.5)

p = vb.DCMTheta(A=A, B=B, C=C)

def dcm(x, up):
    u, p = up
    return vb.dcm_dfun(x, u, p)

_, loop = vb.make_ode(0.2, dcm)
ts = jp.r_[:64]
x0 = jp.ones(5)

conditions = [
    jp.r_[0, 0], # rest
    jp.r_[1, 0], # condition 1
    jp.r_[0, 1], # condition 2
    ]
titles = 'rest,osc 1 2,fast osc 3 4 shift 5'.split(',')
pl.figure(figsize=(8, 8))
for i, cond in enumerate(conditions):
    pl.subplot(1, 3, i + 1)
    xs = loop(x0, ts, (cond, p))
    pl.plot(xs + jp.r_[:n], 'k')
    pl.grid(1)
    pl.title(titles[i])
    pl.xlabel('time (au)')
pl.tight_layout()

# let's try to optimize 

xs_c0 = loop(x0, ts, (conditions[0], p))
xs_c1 = loop(x0, ts, (conditions[1], p))
xs_c2 = loop(x0, ts, (conditions[2], p))

def loss(Bhat):
    p_hat = vb.DCMTheta(A=A, B=Bhat, C=C)
    xs_hat_c0 = loop(x0, ts, (conditions[0], p_hat))
    xs_hat_c1 = loop(x0, ts, (conditions[1], p_hat))
    xs_hat_c2 = loop(x0, ts, (conditions[2], p_hat))
    sse = lambda x,y: jp.sum(jp.square(x-y))
    return sse(xs_hat_c0, xs_c0) + sse(xs_hat_c1, xs_c1) + sse(xs_hat_c2, xs_c2)

vgloss = jax.jit(jax.value_and_grad(loss))
Bhat = jp.zeros((n, n, 2)) + 1e-3
print('initial loss', vgloss(Bhat)[0])
for i in (pbar := tqdm.trange(9000)):
    v, g = vgloss(Bhat)
    Bhat = Bhat - 1e-2*g
    if i%10 == 0:
        pbar.set_description(f'v {v:0.2f} nh {jp.linalg.norm(g):0.2f}')
print('final loss', vgloss(Bhat)[0])

# now let's invert the model for B
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

Bhat0 = Bhat
def logp(xs_c0, xs_c1, xs_c2):
    Bhat = numpyro.sample('Bhat', dist.Normal(Bhat0, 0.1))
    p_hat = vb.DCMTheta(A=A, B=Bhat, C=C)
    xs_hat_c0 = loop(x0, ts, (conditions[0], p_hat))
    xs_hat_c1 = loop(x0, ts, (conditions[1], p_hat))
    xs_hat_c2 = loop(x0, ts, (conditions[2], p_hat))
    numpyro.sample('xs_hat_c0', dist.Normal(xs_hat_c0, 0.1), obs=xs_c0)
    numpyro.sample('xs_hat_c1', dist.Normal(xs_hat_c1, 0.1), obs=xs_c1)
    numpyro.sample('xs_hat_c2', dist.Normal(xs_hat_c2, 0.1), obs=xs_c2)

nuts_kernel = NUTS(logp)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
rng_key = jax.random.PRNGKey(1106)
mcmc.run(rng_key,
         xs_c0=loop(x0, ts, (conditions[0], p)),
         xs_c1=loop(x0, ts, (conditions[1], p)),
         xs_c2=loop(x0, ts, (conditions[2], p)),
         )

Bhat = mcmc.get_samples()['Bhat'].mean(axis=0)

pl.figure()
pl.subplot(321); pl.imshow(B[...,0])
pl.subplot(322); pl.imshow(B[...,1])
pl.subplot(323); pl.imshow(Bhat0[...,0])
pl.subplot(324); pl.imshow(Bhat0[...,1])
pl.subplot(325); pl.imshow(Bhat[...,0])
pl.subplot(326); pl.imshow(Bhat[...,1])
pl.suptitle('B vs Bhat')


pl.show()
