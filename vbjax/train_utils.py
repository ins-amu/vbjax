import jax
import jax.numpy as jnp
from flax import linen as nn

def log_likelihood(ms, logp, x):
   u = jnp.exp(.5 * logp) * (x - ms)
   return -0.5 * (x.shape[0] * jnp.log(2 * jnp.pi) + jnp.sum(u ** 2 - logp, axis=1))


def eval_MADE(made, params, batch, key, shape=None):
  shape = shape if shape else batch.shape
  def eval_model(made):
    ms, logp = made(batch)
    loss = -log_likelihood(ms, logp, batch).mean()
    u_sample = made.gen(key, shape)
    return loss, u_sample
  return nn.apply(eval_model, made)({'params': params})


def train_step(state, batch):
  def loss_fn(params):
    ms, logp = state.apply_fn(
        {'params': params}, batch,
    )
    loss = -log_likelihood(ms, logp, batch).mean()
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


def grad_func(state, batch):
  def loss_fn(params):
    ms, logp = state.apply_fn(
        {'params': params}, batch,
    )
    loss = -log_likelihood(ms, logp, batch).mean()
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return grads
