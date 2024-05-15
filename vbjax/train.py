from flax.training import train_state
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from vbjax.ml_models import GaussianMADE, MAF
import optax
import matplotlib.pyplot as plt
import numpy as np
from vbjax.train_utils import eval_model, train_step, grad_func, log_likelihood_MADE

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

    print(state.params['mlp']['hidden_0']['kernel'][0,:])

    for i, epoch in enumerate(range(config['num_epochs'])):
        if i%10==0:
          ms, logp = model.apply({'params': state.params}, X)
          loss, u_distro = eval_model(
            model, state.params, X, key, log_likelihood_MADE, shape=(20000,2),
          )
          print('eval epoch: {}, loss: {}'.format(i + 1, loss))
          U = jnp.exp(.5 * logp) * (X - ms)
          L = -0.5 * (X.shape[0] * jnp.log(2 * jnp.pi) + jnp.sum(U ** 2 - logp, axis=1))
          fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))
          ax1.scatter(np.array(X[:,0]), np.array(X[:,1]))
          cs = ax2.scatter(np.array(U[:,0]), np.array(U[:,1]), c=L, vmin=L.min(), vmax=L.max())
          his = ax3.hist2d(np.array(u_distro[:,0]), np.array(u_distro[:,1]), bins=30, density=True, vmax=.03)
          fig.colorbar(his[3], ax=ax3)
          ax3.set_title('logp')
          plt.colorbar(cs, ax=ax2)
          fig.tight_layout()
          plt.show()
        for _ in range(100):
          batch = X[i*config['batch_size']:(i+1)*config['batch_size']]
          rng, key = random.split(rng)
          state = train_step(state, batch, log_likelihood_MADE)
        
        ms, logp = model.apply({'params': state.params}, batch)
        loss, u_distro = eval_model(
        model, state.params, batch, key, log_likelihood_MADE
        )
        print('eval epoch: {}, loss: {}'.format(i + 1, loss))
        # print(jnp.linalg.norm(grad_func(state, batch)['mlp']['hidden_0']['kernel']))

    return state, model



config = {}
config['learning_rate'] = .01
config['in_dim'] = 2
config['batch_size'] = 256
config['num_epochs'] = 120
config['n_hiddens'] = [5, 5]


key1, key2 = random.split(random.key(0), 2)
x2 = 4 * random.normal(key1, (config['batch_size']*100,))
x1 = (.25*x2**2) + random.normal(key2, (config['batch_size']*100,))
X = jnp.vstack([x2, x1]).T

# key1, key2 = random.split(random.key(0), 2)
# x2 = 3 * random.normal(key1, (config['batch_size']*100,))
# x1 = x2 + random.normal(key2, (config['batch_size']*100,))
# X = jnp.vstack([x2, x1]).T


model=GaussianMADE(random.PRNGKey(42), 2, config['n_hiddens'], act_fn=nn.relu)
state_f, mdl = train_and_evaluate(model, X, config)


