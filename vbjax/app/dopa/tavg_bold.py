#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
# import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb


# In[2]:


# start with default parameters        
params = vb.dopa_default_theta
params


# let's do one simulation on that just to see what it's like,

# In[3]:


dt = 0.001
sigma = 1e-3
T = 256.0 # ms
n_nodes = 8
Ci,Ce,Cd = vb.randn(3, n_nodes, n_nodes)
key = jax.random.PRNGKey(42)
nt = int(T / dt)


# In[4]:


_, loop = vb.make_sde(dt, vb.dopa_net_dfun, sigma)
dW = jax.random.normal(key, (nt, 6, n_nodes))


# In[5]:


init = jp.array([0.1, -70.0, 0.0, 0.0, 0.0, 0.0])
init = jp.outer(init, jp.ones(n_nodes))

ys = loop(init, dW, (Ci,Ce,Cd,params))
ys.shape


# In[6]:


figure(figsize=(7,1)); plot(ys[:,0,0], 'kx'); grid(1)


# let's make a time average,

# In[7]:


ta, ta_step, ta_sample = vb.make_timeavg(init.shape)
ta_sample = vb.make_offline(step_fn=ta_step, sample_fn=ta_sample)


# since we have several pieces now, put into a dictionary to keep track,

# In[8]:


sim = {
    'ta': ta,
    'init': init,
    'p': (Ci, Ce, Cd, params),
}
# chunk length == 1ms
nt = int(1.0 / dt)

def sim_step(sim, key):
    dW = jax.random.normal(key, (nt, 6, n_nodes))
    ys = loop(sim['init'], dW, sim['p'])
    sim['ta'], ta_y = ta_sample(sim['ta'], ys)
    sim['init'] = ys[-1]
    return sim, ta_y

sim, ta_y = sim_step(sim, key)
ta_y.shape


# we can put this stepping function into a loop like so,

# In[9]:


nT = 10 # number of calls to `sim_step`
keys = jax.random.split(key, nT)
sim, ta_y = jax.lax.scan(sim_step, sim, keys)
ta_y.shape


# In[10]:


plot(ta_y[:,0], 'k')


# we could wonder whether this continuation of the simulation is working correctly?
# 
# let's write a function to do the simulation at different temporal averaging periods, for the same RNG key and initial conditions (i.e. exact same simulation):

# In[11]:


def run_with_tavg(init, key, tavg_period, total_time=100.0):
    
    sim = {
        'ta': ta,
        'init': init,
        'p': (Ci, Ce, Cd, params),
    }
    
    # tavg period
    nt = int(tavg_period / dt)
    
    def sim_step(sim, dw):
        ys = loop(sim['init'], dw, sim['p'])
        sim['ta'], ta_y = ta_sample(sim['ta'], ys)
        sim['init'] = ys[-1]
        return sim, ta_y

    nta = int(np.ceil(total_time / tavg_period))
    ts = np.r_[:nta]*tavg_period
    dw = jax.random.normal(key, (ts.size * nt, 6, n_nodes)).reshape(ts.size,nt,6,n_nodes)
    sim, ta_y = jax.lax.scan(sim_step, sim, dw)
    return ts, ta_y

t, y = run_with_tavg(init, key, 1.0)
t.shape, y.shape


# now let's run it for three different sample periods with dt as a baseline, and observe how, for deterministic flow, the trajectories are mostly the same across different temporal averaging periods, but when the system enters fixed point stochastic fluctuations, temporal average diverages. 

# In[12]:


init = jp.array([0.035, -67.0, 0, 0.0, 0.0, 0.0])
init = jp.outer(init, jp.ones(n_nodes))

for tap in [dt, 32*dt, 256*dt]:
    t, y = run_with_tavg(init, key, tap, total_time=50.0)
    plot(t, y[:,0,0], 'x' if tap>dt else '-', label=f'{tap}')
    
grid(1); legend()


# what if we want temporal average & bold?
# 
# - do temporal average as above
# - put temporal average samples into bold in online fashion
# 
# while the function does everything inside it, in practice we'll split it up to make it easier to read,

# In[ ]:


def run_with_tavg_bold(init, key, tavg_period, total_time=100.0):
    
    bold_buf, bold_step, bold_samp = vb.make_bold(
        shape=init[0].shape, # only r
        dt=tavg_period/1e3,
        p=vb.bold_default_theta)

    sim = {
        'ta': ta,
        'bold': bold_buf,
        'init': init,
        'p': (Ci, Ce, Cd, params),
    }
    
    # tavg period
    nt = int(tavg_period / dt)
    
    def sim_step(sim, dw):
        ys = loop(sim['init'], dw, sim['p'])
        sim['ta'], ta_y = ta_sample(sim['ta'], ys)
        sim['bold'] = bold_step(sim['bold'], ta_y[0])
        _, bold_t = bold_samp(sim['bold'])
        sim['init'] = ys[-1]
        return sim, (ta_y, bold_t)

    nta = int(np.ceil(total_time / tavg_period))
    ts = np.r_[:nta]*tavg_period
    dw = jax.random.normal(key, (ts.size * nt, 6, n_nodes)).reshape(ts.size,nt,6,n_nodes)
    sim, (ta_y, bold) = jax.lax.scan(sim_step, sim, dw)
    return ts, ta_y, bold

t, y, b = run_with_tavg_bold(init, key, 1., total_time=15e3)


# In[ ]:


y.shape, b.shape


# In[ ]:


plot(t, y[:,0,0], 'k', label='tavg r')
plot(t, b[:,0], 'b', label='bold r')
legend(); grid(1)


# In[ ]:


get_ipython().run_line_magic('timeit', 'run_with_tavg_bold(init, key, 1., total_time=100)')


# can we make that more jit-able? it mainly requires "lifting" the integers, so that for the function we call `jit` on, those integers are "compile" time constants:

# In[128]:


def make_run_tavg_bold(tavg_period, total_time=100.0):
    nt = int(tavg_period / dt)
    nta = int(np.ceil(total_time / tavg_period))
    
    def run(init, key):
        
        bold_buf, bold_step, bold_samp = vb.make_bold(
            shape=init[0].shape, # only r
            dt=tavg_period/1e3,
            p=vb.bold_default_theta)
    
        sim = {
            'ta': ta,
            'bold': bold_buf,
            'init': init,
            'p': (Ci, Ce, Cd, params),
        }
        
        def sim_step(sim, dw):
            ys = loop(sim['init'], dw, sim['p'])
            sim['ta'], ta_y = ta_sample(sim['ta'], ys)
            sim['bold'] = bold_step(sim['bold'], ta_y[0])
            _, bold_t = bold_samp(sim['bold'])
            sim['init'] = ys[-1]
            return sim, (ta_y, bold_t)
    
        ts = np.r_[:nta]*tavg_period
        dw = jax.random.normal(key, (ts.size * nt, 6, n_nodes)).reshape(ts.size,nt,6,n_nodes)
        sim, (ta_y, bold) = jax.lax.scan(sim_step, sim, dw)
        return ts, ta_y, bold

    return jax.jit(run)

r = make_run_tavg_bold(1., total_time=100.0)
r(init, key)
get_ipython().run_line_magic('timeit', 'r(init, key)')


# gives us a nice 10x improvement.  that is btw,

# In[130]:


int(100.0/dt) / 0.0276


# about 3M step/s.  that is relatively fast given the amount of overhead going into the simulation

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




