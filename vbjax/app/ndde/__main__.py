#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
import jax
import jax.numpy as jp
import numpy as np
import tqdm
import vbjax as vb
from jax.example_libraries.optimizers import adam
import time, pickle


# In[2]:


def load_data(C, M, B, F):
    meg00 = np.load('../../Downloads/megmodes-100307.npy').astype('f')[:,:M] / 1e-5
    meg00.shape
    W = C+F
    meg00wins = jp.array( meg00[:meg00.shape[0] // W * W].reshape((-1, W, M)) )
    print('wins shape and MB', meg00wins.shape, meg00wins.nbytes >> 20)
    ntrain = len(meg00wins)//5*4
    tr_x = meg00wins[:ntrain]
    te_x = meg00wins[ntrain:]
    print('tr te shapes', tr_x.shape, te_x.shape)
    return tr_x, te_x

def load_meg(M):
    meg = jp.load('../../Downloads/megmodes-100307.npy')[1024:-1024,:M]/1e-5
    tr = meg[:meg.shape[0]//5*4]
    te = meg[tr.shape[0]:]
    return tr, te

def gen_ixs(part, C, F, B):
    # np much faster than jax for this
    import numpy as np
    nwin = part.shape[0] // F + B
    nwin = nwin // B * B
    lastt = part.shape[0] - (C + F)
    ix = np.random.randint(0, lastt, (nwin,))
    return jp.array(np.array(np.split(ix, ix.size // B)))

def make_get_batch(C, F):
    @jax.jit
    def get_batch(part, ix):
        get1 = lambda i: jax.lax.dynamic_slice_in_dim(part, i, C+F)
        return jax.vmap(get1)(ix)
    return get_batch


# In[3]:


def make_dense_layers(in_dim, latent_dims=[10], out_dim=None, init_scl=0.1, extra_in=0,
               act_fn=jax.nn.leaky_relu, key=jax.random.PRNGKey(42)):
    """Make a dense neural network with the given latent layer sizes."""
    small = lambda shape: jax.random.normal(key, shape=shape)*init_scl
    # flops of n,p x p,m is nm(2p-1)
    if len(latent_dims) > 0:
        weights = [small((latent_dims[0], in_dim + extra_in))]
        biases = [small((latent_dims[0], ))]
    else:
        print('no latent dims for mlp: linear model only!')
        weights, biases = [], []
    nlayers = len(latent_dims)
    
    for i in range(nlayers - 1):
        weights.append(small((latent_dims[i+1], latent_dims[i])))
        biases.append(small((latent_dims[i+1], )))
        
    weights.append(small((out_dim or in_dim, latent_dims[-1] if latent_dims else in_dim)))
    biases.append(small((out_dim or in_dim, )))
    
    def fwd(params, x):
        weights, biases = params
        for i in range(len(weights) - 1):
            x = act_fn(weights[i] @ x + biases[i])
        return weights[-1] @ x + biases[-1]

    return (weights, biases), fwd


def make_run1(mlp, forecast, C, F):

    def run1(wb, x):
    
        def op(ctx, y):
            cx = mlp(wb, ctx.ravel())
            nx = ctx[-1] + 0.1*(-ctx[-1] + cx)
            ctx = jp.roll(ctx, shift=-1, axis=0)
            ctx = ctx.at[-1].set(nx)
            return ctx, nx if forecast else jp.mean(jp.square(nx - y))

        y = jp.r_[:F] if forecast else x[C:]
        ctx, outs = jax.lax.scan(op, x[:C], y)
        return outs
    return run1

def make_loss(runb_loss):
    def loss(wb, x):
        losses = runb_loss(wb, x)
        return jp.mean(losses)
    return [jax.jit(_) for _ in (loss, jax.grad(loss), jax.value_and_grad(loss))]


# In[4]:


def save_params(i, wb):
    with open(f'/tmp/megnode-v2-{i}.pkl', 'wb') as fd:
        pickle.dump(wb, fd)


# In[5]:


def __make_batch_ix(n, b):
    ix = np.r_[:n]
    np.random.shuffle(ix)
    ixs = np.array_split(ix, n // b + 1)
    return [jp.array(_) for _ in ixs]


# In[6]:


def train(i, tr_x, C, F, loss_and_grads, get_batch, aupdate, aget, awb, batch):
    train_loss = []
    for ix in gen_ixs(tr_x, C, F, batch):
        bx = get_batch(tr_x, ix)
        tr_loss, tr_grads = loss_and_grads(aget(awb), bx)
        train_loss.append(tr_loss)
        awb = aupdate(i, tr_grads, awb)
    train_loss = np.mean(train_loss)
    return awb, train_loss

def test(te_x, C, F, loss, get_batch, aget, awb, batch):
    test_loss = []
    for ix in gen_ixs(te_x, C, F, batch):
        bx = get_batch(te_x, ix)
        te_loss = loss(aget(awb), bx)
        test_loss.append(te_loss)
    test_loss = np.mean(test_loss)
    return test_loss

def run_epochs(wb, tr_x, te_x, C, F, loss, loss_and_grads, get_batch,
               niter=200, sched='linear', lr_base=1e-4, batch=2048, patience=3600):
    trace = []
    ainit, _, aget = adam(lr_base)
    awb = ainit(wb)
    tik0 = time.time()
    for i in (pbar := tqdm.trange(niter, ncols=80)):
        try:
            tik = time.time()
            lr = lr_base
            if sched == 'linear':
                lr = lr / (1 + i/niter)
            _, aupdate, _ = adam(lr)
            # train + test
            awb, train_loss = train(i, tr_x, C, F, loss_and_grads, get_batch, aupdate, aget, awb, batch)
            test_loss = test(te_x, C, F, loss, get_batch, aget, awb, batch)
            # progress
            msg = f'{lr:0.2g} {train_loss:0.2f} {test_loss:0.2f}'
            pbar.set_description(msg)
            if i % 10 == 0:
                save_params(i, aget(awb))
            tok = time.time()
            trace.append([lr, train_loss, test_loss, tok-tik])
            
        except KeyboardInterrupt:
            print('stopping early')
            break
        if (time.time() - tik0) > patience:
            break
    
    return aget(awb), np.array(trace).T


# In[7]:


def run(C, F, M, B=2048, A=[256,256], lr=1e-4, patience=300, niter=2000, continue_wb=None):
    get_batch = make_get_batch(C, F)
    tr_x, te_x = load_meg(M)
    wb, mlp = make_dense_layers(C*M, A, out_dim=M, init_scl=1e-5)
    if continue_wb is not None:
        wb = continue_wb
    run1_loss = make_run1(mlp, forecast=False, C=C, F=F)
    runb_loss = lambda wb, x: jax.vmap(lambda x: run1_loss(wb, x))(x)
    loss, gloss, loss_and_grads = make_loss(runb_loss)
    ixs = gen_ixs(te_x, C, F, B)
    bx = get_batch(te_x, ixs[0])
    loss(wb, bx)
    return run_epochs(wb, tr_x, te_x, C, F, loss, loss_and_grads, get_batch, niter=niter, batch=B, lr_base=lr*(B/512), patience=patience)

#wb, (lr, trl, tel, tt) = run(4, 1, 4, patience=10)


# In[8]:


def plot_losses(trl, tel):
    import pylab as pl
    pl.loglog(trl, '-', label='train loss', color='k', alpha=0.7)
    pl.loglog(tel, '--', label='test loss', color='r', alpha=0.7)
    pl.grid(1); pl.legend(); pl.xlabel('epoch'); pl.ylabel('loss')


# - try out different batch-lr relations.  sqrt(k) seems to favor small batches, but lower performance.  This seems to remain true for large MLPs too.
# - C32, F8, A 8192,4096,2048; lr_base 1e-4: tr gets down to 10 but te goes up, overfitting
# - C128, F32, A 2048x3: still overfitting after 100 epochs
# - C=F=128 B=2048 A=1024,256 no overfit but slow learning
# - random data sampling makes minibatch loss much more jumpy
# - a sweep over C, 8, 16, A=`[1024,256]` shows overfitting tendency with larger C which is also larger arch
# - first layer of `C*M x A[0]` represents largest layer by far: this is low hanging fruit for improvement
# - sweep over C with F=32 shows no overfitting, even for larger C

# In[ ]:


#wb = run(16, 4, 16, A=[256,256], patience=3600, lr=1e-4)[0]
M = 96
F = 128
C = 128

# simple architecture keeps it easily interpretable
# even 4*M, overfits!
A = [4*M, M//4, 4*M]  # overfits around 2e4 iterations
A = [2*M]

wb, (_, trl, tel, _) = run(C, F, M, B=2048, A=A, patience=99999, niter=10000, lr=1e-5)  #, continue_wb=wb)



# In[ ]:


len(wb[0])


# In[13]:


plot_losses(trl, tel)


# In[ ]:


wb2, (_, trl2, tel2, _) = run(C, F, M, B=4096, A=A, patience=99999, niter=50000, lr=1e-5, continue_wb=wb)




# In[ ]:


plot_losses(trl2, tel2)


# In[ ]:


w0 = wb[0][0]
w0.shape


# In[ ]:


w0_ = w0.reshape(A[0], C, M)
w0_.shape


# In[ ]:


w0_[25]


# In[ ]:


imshow(w0_[45].T, vmin=-0.1, vmax=0.1)


# could achieve some fourier basis or just use a cnn?
# 
# the above fits `A[0], C, M` which is like `A[0]*M` filters or each `M` getting `A[0]` filters, with a subsequent sum over `M`. 
# 
# but why not 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




