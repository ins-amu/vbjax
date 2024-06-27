from flax.training import train_state
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from vbjax.ml_models import GaussianMADE, MAF
import optax, jax
import matplotlib.pyplot as plt
import numpy as np
from vbjax.train_utils import eval_model, grad_func, log_likelihood_MAF

def train_step(state, batch, loss_f):
  def loss_fn(params):
    output = state.apply_fn(
        {'params': params}, batch,
    )
    loss = loss_f(output, batch).mean()
    return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)


def train_and_evaluate(model, X, config):
    rng = random.key(0)
    rng, key = random.split(rng)

    init_data = jnp.ones((config['batch_size'], config['in_dim']), jnp.float32)
    params = model.init(key, init_data)['params']

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(config['learning_rate']),
    )

    # print(state.params['mlp']['hidden_0']['kernel'][0,:])
    batch_size = config['batch_size']
    BATCHES = np.split(np.random.choice(np.arange(len(X)), (len(X)//batch_size)*batch_size), (len(X)//batch_size))

    for i, epoch in enumerate(range(config['num_epochs'])):
        if i%10==0:
          u, logp = state.apply_fn({'params': state.params}, X)
          loss, u_distro = eval_model(
            model, state.params, X, key, log_likelihood_MAF, shape=(20000,2),
          )
          print('eval epoch: {}, loss: {}'.format(i + 1, loss))
          # U = jnp.exp(.5 * logp) * (X - ms)
          # L = -0.5 * (X.shape[1] * jnp.log(2 * jnp.pi) + jnp.sum(U ** 2 - logp, axis=1))
          fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))
          ax1.scatter(np.array(X[:,0]), np.array(X[:,1]))
          cs = ax2.scatter(np.array(u[:,0]), np.array(u[:,1]), c=loss, vmin=loss.min(), vmax=loss.max())
          his = ax3.hist2d(np.array(u_distro[:,0]), np.array(u_distro[:,1]), bins=30, density=True, vmax=.03)
          fig.colorbar(his[3], ax=ax3)
          ax3.set_title('logp')
          plt.colorbar(cs, ax=ax2)
          fig.tight_layout()
          plt.show()

        for j, batch_i in enumerate(BATCHES):
          batch = X[batch_i]
          # batch = X[i*config['batch_size']:(i+1)*config['batch_size']]
          rng, key = random.split(rng)
          state = train_step(state, batch, log_likelihood_MAF)
        loss, u_distro = eval_model(
            model, state.params, X, key, log_likelihood_MAF, shape=(20000,2),
          )
        print('eval epoch: {}, loss: {}'.format(i + 1, loss.mean()))
        # print(jnp.linalg.norm(grad_func(state, batch)['mlp']['hidden_0']['kernel']))

    return state, model



config = {}
config['learning_rate'] = .003
config['in_dim'] = 2
config['batch_size'] = 256
config['num_epochs'] = 60
config['n_hiddens'] = [5, 5]


key1, key2 = random.split(random.key(0), 2)
x2 = 4 * random.normal(key1, (config['batch_size']*100,))
x1 = (.25*x2**2) + random.normal(key2, (config['batch_size']*100,))
X = jnp.vstack([x2, x1]).T

# key1, key2 = random.split(random.key(0), 2)
# x2 = 3 * random.normal(key1, (config['batch_size']*100,))
# x1 = x2 + random.normal(key2, (config['batch_size']*100,))
# X = jnp.vstack([x2, x1]).T


model=MAF(random.PRNGKey(42), 2, config['n_hiddens'], act_fn=nn.relu, n_mades=4)
state_f, mdl = train_and_evaluate(model, X, config)


def nnet_fn(X):
  ms, logp = mdl.apply({'params': state_f.params}, X)
  u = jnp.exp(.5 * logp) * (X - ms)
  L = -0.5 * (X.shape[1] * jnp.log(2 * jnp.pi) + jnp.sum(u ** 2 - logp, axis=1))
  return u, L


x = y = jnp.linspace(-30, 30, 500)
xx, yy = jnp.meshgrid(x, y)
X_test = jnp.vstack([xx.ravel(), yy.ravel()]).T
u, L = nnet_fn(X_test)
L = L.reshape(500,500)
plt.imshow(np.exp(L))
plt.show()