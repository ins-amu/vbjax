import jax
import jax.numpy as jnp
from flax import linen as nn

def log_likelihood_MADE(ms, logp, x, *args):
   u = jnp.exp(.5 * logp) * (x - ms)
   return -(- 0.5 * (x.shape[1] * jnp.log(2 * jnp.pi) + jnp.sum(u ** 2 - logp, axis=1)))


def log_likelihood_MAF(x, *arg):
    u, logdet_dudx = x
    return -(- 0.5 * u.shape[1] * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(u ** 2, axis=1) + logdet_dudx)

def mse_ode(traj, x, *arg):
  return jnp.mean((traj-x)**2)


def eval_model_ode(model, params, batch, loss_fn=None, shape=None):
  batch, i_ext = batch
  def eval_model(model):
    output = model(batch, i_ext)
    loss = loss_fn(output, batch)
    return loss
  return nn.apply(eval_model, model)({'params': params})


def eval_loss(model, params, batch, loss_fn=None, shape=None):
  batch, i_ext = batch
  def eval_model(model):
    output = model(batch, i_ext)
    loss = loss_fn(output, batch)
    return loss
  return nn.apply(eval_model, model)({'params': params})

def eval_model(model, params, batch, key, loss_fn, shape=None):
  shape = shape if shape else batch.shape
  def eval_model(model):
    output = model(batch)
    loss = loss_fn(output, batch)#.mean()
    u_sample = model.gen(key, shape)
    return loss, u_sample
  return nn.apply(eval_model, model)({'params': params})


def train_step(state, batch, loss_f):
  batch, p = batch
  def loss_fn(params):
    output = state.apply_fn(
        {'params': params}, batch, p,
    )
    loss = loss_f(output, batch)
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


def grad_func(state, batch, loss_fn):
  def loss_fn(params):
    output = state.apply_fn(
        {'params': params}, batch,
    )
    loss = loss_fn(output, batch)
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return grads

def loss_t(traj, X):
    X, iext = X
    loss_bias  = jnp.var(X, axis=2)*10+1
    squared_loss_vec = jnp.square(X - traj).mean(axis=(2))
    return (loss_bias*squared_loss_vec).sum()

def loss_t_unpack(traj, X):
    X, iext = X
    loss_bias  = jnp.var(X, axis=2)*40+1
    squared_loss_vec = jnp.square(X - traj).mean(axis=(2))
    return (loss_bias*squared_loss_vec).sum(axis=1)