import jax
import jax.numpy as jnp
from flax import linen as nn

def log_likelihood_MADE(ms, logp, x, *args):
   u = jnp.exp(.5 * logp) * (x - ms)
   return -0.5 * (x.shape[1] * jnp.log(2 * jnp.pi) + jnp.sum(u ** 2 - logp, axis=1))


def log_likelihood_MAF(u, logdet_dudx, *arg):
    return -0.5 * u.shape[1] * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(u ** 2, axis=1) + logdet_dudx


def eval_model(model, params, batch, key, likelihood_fn, shape=None):
  shape = shape if shape else batch.shape
  def eval_model(model):
    output = model(batch)
    loss = -likelihood_fn(*output, batch).mean()
    u_sample = model.gen(key, shape)
    return loss, u_sample
  return nn.apply(eval_model, model)({'params': params})


def train_step(state, batch, likelihood_fn):
  def loss_fn(params):
    output = state.apply_fn(
        {'params': params}, batch,
    )
    loss = -likelihood_fn(*output, batch).mean()
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


def grad_func(state, batch, likelihood_fn):
  def loss_fn(params):
    output = state.apply_fn(
        {'params': params}, batch,
    )
    loss = -likelihood_fn(*output, batch).mean()
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return grads
