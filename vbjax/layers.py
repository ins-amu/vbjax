import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence
from jaxlib.xla_extension import ArrayImpl
import jax.random as random


def make_dense_layers(in_dim, latent_dims=[10], out_dim=None, init_scl=0.1, extra_in=0,
               act_fn=jax.nn.leaky_relu, key=jax.random.PRNGKey(42)):
    """Make a dense neural network with the given latent layer sizes."""
    small = lambda shape: jax.random.normal(key, shape=shape)*init_scl
    # flops of n,p x p,m is nm(2p-1)
    weights = [small((latent_dims[0], in_dim + extra_in))]
    biases = [small((latent_dims[0], 1))]
    nlayers = len(latent_dims)
    
    for i in range(nlayers - 1):
        weights.append(small((latent_dims[i+1], latent_dims[i])))
        biases.append(small((latent_dims[i+1], 1)))
        
    weights.append(small((out_dim or in_dim, latent_dims[-1])))
    biases.append(small((out_dim or in_dim, 1)))
    
    def fwd(params, x):
        weights, biases = params
        for i in range(len(weights) - 1):
            x = act_fn(weights[i] @ x + biases[i])
        return weights[-1] @ x + biases[-1]
    
    return (weights, biases), fwd

def create_degrees(key, n_inputs, n_hiddens, input_order, mode):
    """
    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :return: list of degrees
    """

    degrees = []

    # create degrees for inputs
    if isinstance(input_order, str):

        if input_order == 'random':
            degrees_0 = jnp.arange(1, n_inputs + 1)
            jax.random.permutation(key, degrees_0)

        elif input_order == 'sequential':
            degrees_0 = jnp.arange(1, n_inputs + 1)

        else:
            raise ValueError('invalid input order')

    else:
        input_order = jnp.array(input_order)
        assert jnp.all(jnp.sort(input_order) == jnp.arange(1, n_inputs + 1)), 'invalid input order'
        degrees_0 = input_order
    degrees.append(degrees_0)

    # create degrees for hiddens
    if mode == 'random':
        for N in n_hiddens:
            min_prev_degree = min(jnp.min(degrees[-1]), n_inputs - 1)
            degrees_l = jax.random.randint(key, shape=(N,), minval=min_prev_degree, maxval=n_inputs)
            degrees.append(degrees_l)

    elif mode == 'sequential':
        for N in n_hiddens:
            degrees_l = jnp.arange(N) % max(1, n_inputs - 1) + min(1, n_inputs - 1)
            degrees.append(degrees_l)

    else:
        raise ValueError('invalid mode')

    return degrees


def create_masks(degrees):
    """
    Creates the binary masks that make the connectivity autoregressive.
    :param degrees: a list of degrees for every layer
    :return: list of all masks, as theano shared variables
    """

    Ms = []

    for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        M = d0[:, jnp.newaxis] <= d1
        # M = theano.shared(M.astype(dtype), name='M' + str(l+1), borrow=True)
        Ms.append(M)

    Mmp = degrees[-1][:, jnp.newaxis] < degrees[0]
    # Mmp = theano.shared(Mmp.astype(dtype), name='Mmp', borrow=True)

    return Ms, Mmp


class MaskedLayer(nn.Module):
  features: int
  mask: ArrayImpl
  kernel_init: Callable = lambda key, shape, mask: random.normal(key, shape=shape)*mask
  bias_init: Callable = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel',
                        self.kernel_init, # Initialization function
                        (inputs.shape[-1], self.features), self.mask)  # shape info.
    y = jnp.dot(inputs, kernel*self.mask)
    bias = self.param('bias', self.bias_init, (self.features,))
    y = y + bias
    return y


class OutputLayer(nn.Module):
  out_dim: int
  out_mask: ArrayImpl
  kernel_init: Callable = lambda key, shape, out_mask: jax.random.normal(key, shape=shape)/jnp.sqrt(shape[0])*out_mask
  bias_init: Callable = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs):
    kernel_m = self.param('kernel_m',
                        self.kernel_init, # Initialization function
                        (inputs.shape[-1], self.out_dim), self.out_mask)  # shape info.
    kernel_logp = self.param('kernel_logp',
                        self.kernel_init, # Initialization function
                        (inputs.shape[-1], self.out_dim), self.out_mask)
    m = jnp.dot(inputs, kernel_m*self.out_mask)
    logp = jnp.dot(inputs, kernel_logp*self.out_mask)
    bias_m = self.param('bias_m', self.bias_init, (self.out_dim,))
    bias_logp = self.param('bias_logp', self.bias_init, (self.out_dim,))
    m = m + bias_m
    logp = logp + bias_logp
    return m, logp


class MaskedMLP(nn.Module):
    n_hiddens: Sequence[int]
    act_fn: Callable
    masks: Sequence[ArrayImpl]

    def setup(self):
        self.hidden = [MaskedLayer(mask.shape[1], mask) for mask in self.masks]
    
    def __call__(self, inputs):
        x = inputs
        for i, (layer,) in enumerate(zip(self.hidden)):
            x = layer(x)
            # if i != len(self.hidden) - 1:
            x = self.act_fn(x)
        return x



