#!/usr/bin/env python
# coding: utf-8

# In[1]:

try:
    get_ipython
except:
    raise RuntimeError('please run this script with ipython! (or open a PR to clean up the script :)')


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import jax.numpy as jp
import jax
import vbjax as vb
import tqdm
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'jpeg'")


# In[2]:


def dfun1(x, p):
    c, p = p
    return vb.mpr_dfun(x, (c,0), p)

dt = 0.1
nt = 200
ts = jp.r_[:nt]
_, loop = vb.make_ode(dt, dfun1)

def run_it(pars):
    c, tau, r0 = pars
    rv0 = jp.r_[r0, -2.0]
    p = c, vb.mpr_default_theta._replace(tau=tau)
    rvs = loop(rv0, ts, p)
    return rvs

run_them = jax.jit(jax.vmap(run_it))
ng = 4j
cs,taus,r0s=jp.mgrid[0.0:2.0:ng, 1.0:3.0:ng, 0.001:1.0:ng]
pars = jp.c_[cs.ravel(), taus.ravel(), r0s.ravel()]
rvs = run_them(pars)

figure(figsize=(5,2)); plot(rvs[-15]); grid(1); print(pars[-15])


# In[3]:


def sample_pars(key, batch_size):
    c,tau,r0 = jax.random.uniform(key, (batch_size, 3)).T
    c = c * 2
    tau = (tau * 2) + 1
    r0 = r0 + 0.001
    return jp.c_[c, tau, r0]

sample_pars(jax.random.PRNGKey(42), 4)


# In[4]:


wb, f = vb.make_dense_layers(in_dim=2, extra_in=2, latent_dims=[32]*3, init_scl=1e-1)

def dfun2(x, p):
    r, V = x
    wb, (c, tau) = p
    x = jp.r_[r, V, c, tau].reshape(-1, 1)
    dx = f(wb, x).reshape(2)
    return dx

# let's try an intermediate step to fit the derivatives

def dfun1_(rv, cp):
    return dfun1(rv, (cp[0], vb.mpr_default_theta._replace(tau=cp[1])))

def loss(wb, pars):
    rvs = run_them(pars)
    drv1 = jax.vmap(lambda rv_t: jax.vmap(dfun1_)(rv_t, pars), in_axes=1, out_axes=1)( rvs )
    drv2 = jax.vmap(lambda rv_t: jax.vmap(lambda x,p: dfun2(x, (wb, p[:2])))(rv_t, pars), in_axes=1, out_axes=1)( rvs )
    return jp.mean(jp.square(drv1 - drv2))

loss(wb, pars)


# In[5]:


key = jax.random.PRNGKey(42)
loss(wb, sample_pars(key, 4))

vg_loss = jax.jit(jax.value_and_grad(loss))
batch_size = 8**pars.shape[1]
from jax.example_libraries.optimizers import adam
step_size = 1e-2
opt = adam(step_size)
wb1 = opt.init_fn(wb)
trace = []

for i in (pbar := tqdm.trange(2000, ncols=80)):
    pars = sample_pars(key, batch_size)
    v, g = vg_loss(opt.params_fn(wb1), pars)
    
    if not jp.isfinite(v):
        print('broke')
        break
    # wb1 = jax.tree.map(lambda l, gl: l - 0.01*gl, wb1, g)
    
    ng = jax.example_libraries.optimizers.l2_norm(g)
    nw = jax.example_libraries.optimizers.l2_norm(opt.params_fn(wb1))
    wb1 = opt.update_fn(i, g, wb1)
    trace.append((v, ng))
    if i%10 == 0:
        pbar.set_description(f'll {jp.log(v):0.4f} lng {jp.log(ng):0.2f} lnw {jp.log(nw):0.2f}')
    if i%100 == 0:
        key, key_i = jax.random.split(key)


# In[6]:


semilogy(trace[10:], alpha=0.3)


# now let's construct the trajectories for this first optimization pass,

# In[7]:


_, loop2 = vb.make_ode(dt, dfun2)

def run_it2(wb, pars):
    c, tau, r0 = pars
    rv0 = jp.r_[r0, -2.0]
    p = jp.r_[c, tau]
    rvs = loop2(rv0, ts, (wb, p))
    return rvs

test_pars = sample_pars(jax.random.PRNGKey(42), 2)
rvs = run_them(test_pars)
rvs2 = jax.vmap(lambda p: run_it2(opt.params_fn(wb1), p))(test_pars)

figure(figsize=(8, 2)); plot(rvs[:10,:,0].T, 'k'); plot(rvs2[:10,:,0].T, 'r'); grid(1);


# just from fitting derivatives, very close.  perhaps also Jacobians later.
# 
# for now, let's fine tune, gradually fitting more of the time series

# In[8]:


def make_forecast_loss(t0, t1):
    def forecast_loss(wb, pars):
        rvs = run_them(pars)
        rvs2 = jax.vmap(lambda p: run_it2(wb, p))(pars)
        mse_t = jp.mean(jp.square(rvs[...,0] - rvs2[...,0]), axis=0)
        assert mse_t.shape == (200,)
        return jp.mean(mse_t[t0:t1])
    return forecast_loss


# In[9]:


step_size = 1e-4
opt = adam(step_size)
wb2 = opt.init_fn(opt.params_fn(wb1))
forecast_trace = []
vg_loss = [jax.jit(jax.value_and_grad(make_forecast_loss(0, (i+1)*10 ))) for i in range(10)]

for i in range(401):
    key, key_i = jax.random.split(key)
    pars = sample_pars(key_i, batch_size)
    v, g = vg_loss[i//41](opt.params_fn(wb2), pars)
    ng = jax.example_libraries.optimizers.l2_norm(g)
    nw = jax.example_libraries.optimizers.l2_norm(opt.params_fn(wb2))
    wb2 = opt.update_fn(i, g, wb2)
    if i % 100 == 0:
        print(i, v, ng, nw)

test_pars = sample_pars(jax.random.PRNGKey(42), 2)
rvs = run_them(test_pars)
subplot(211); 
rvs2 = jax.vmap(lambda p: run_it2(opt.params_fn(wb1), p))(test_pars)
plot(rvs[...,0].T, 'k'); plot(rvs2[...,0].T, 'r'); grid(1);
subplot(212); 
rvs2 = jax.vmap(lambda p: run_it2(opt.params_fn(wb2), p))(test_pars)
plot(rvs[...,0].T, 'k'); plot(rvs2[...,0].T, 'r'); grid(1);


# that's very close, we could go back to the latent dims to see fi a smaller arch would work.
# 
# in any case, we can now use that to make a network simulation,

# In[10]:


nnode = 90

params = {
    'weights': jp.abs(vb.randn(nnode, nnode)),
    'tau': 1.0,
    'k': 0.01,
    'sigma': 1e-3,
    'wb': wb
}

def dfun2_one(wb, tau, x, c):
    r, V = x
    x = jp.r_[r, V, c, tau].reshape(-1, 1)
    dx = f(wb, x).reshape(2)
    return dx

def dfun2_net(rv, p):
    cr = p['k'] * (p['weights'] @ rv[0]).reshape(1, -1)
    dx = jax.vmap(lambda x, c: dfun2_one(p['wb'], p['tau'], x, c), in_axes=1, out_axes=1)(rv, cr)
    return dx

def gfun_net(_, p):
    return p['sigma']


# In[11]:


_, loop = vb.make_sde(dt, dfun2_net, gfun=gfun_net, unroll=10, adhoc=vb.mpr_r_positive)

@jax.jit
def run_one_second(sim, key):
    def op(sim, key):
        z = vb.randn(100, 2, nnode, key=key)
        rv = loop(sim['rv'], z, sim['params'])
        sim['rv'] = rv[-1]
        return sim, rv.reshape((10, 10, 2, nnode)).mean(axis=1)
    keys = jax.random.split(key, 100) # 100 * 10 ms
    return jax.lax.scan(op, sim, keys)

# pack buffers and run it one minute
sim = {
    'params': params,
    'rv': jp.zeros((2, nnode)) + jp.c_[0.1, -2.0].T,
}
rvs = []
keys = jax.random.split(jax.random.PRNGKey(42), 60)
for i, key in enumerate(tqdm.tqdm(keys, ncols=60)):
    sim, rv = run_one_second(sim, key)
    rvs.append(rv)


# In[12]:


rvs = jp.array(rvs).reshape((-1, 2, nnode))
rvs.shape


# In[13]:


t = np.r_[:len(rvs)]/1e3
plot(t, rvs[:, 0, :], 'k-');
xlabel('time (s)')
ylabel('rate activity')


# Now let's put that into a parameter sweep by writing a function which takes
# a set of parameters and a number of seconds to run the simulation for:

# In[14]:


weights = jp.abs(vb.randn(nnode, nnode))

def run_pars(pars, nt=10):
    tau, k, sigma = pars
    params = {
        'weights': weights,
        'tau': tau, # 1.0,
        'k': k, # 0.01,
        'sigma': sigma, # 1e-3,
        'wb': wb
    }
    sim = {
        'params': params,
        'rv': jp.zeros((2, nnode)) + jp.c_[0.1, -2.0].T,
    }
    keys = jax.random.split(jax.random.PRNGKey(42), nt)
    sim, rvs = jax.lax.scan(run_one_second, sim, keys)
    return rvs.reshape(-1, 2, nnode)


# In[15]:


get_ipython().run_line_magic('timeit', 'run_pars((2.0, 0.1, 0.01), nt=10)')


# We can then `jax.vmap` this over parameter sweep,

# In[16]:


pars = sample_pars(jax.random.PRNGKey(42), 32)
run_sweep = jax.jit(jax.vmap(run_pars))
get_ipython().run_line_magic('timeit', 'run_sweep(pars)')


# On a GPU we can see the time to run 32 or 1 simulation is the same, so vmap works effectively on GPU (contrary to CPU).  We can then test scaling with the size of sweep, results will depend on hardware; this is with an RTX 4090,

# In[17]:


for n in [32, 64, 128, 256, 512, 1024]:
    pars = sample_pars(jax.random.PRNGKey(42), n)
    get_ipython().run_line_magic('timeit', 'run_sweep(pars)')


# Each simulation corresponds to 10 seconds at a dt of 0.1 ms, so 100k iterations per simulation.  In the largest sweep of 1024 simulations, that's 102.4M iterations in ~15s or 

# In[18]:


1024 * 0.1 / 15.2, 'M iter / s'


# This is not as fast as pure CUDA code would be but it's quick enough for prototyping.
