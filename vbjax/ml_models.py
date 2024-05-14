import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence
from jax._src.prng import PRNGKeyArrayImpl
import jax.random as random
from vbjax.layers import MaskedMLP, OutputLayer, create_degrees, create_masks


class GaussianMADE(nn.Module):
    key: PRNGKeyArrayImpl
    in_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable

    def setup(self):
        self.degrees = create_degrees(self.key, self.in_dim, self.n_hiddens, input_order='sequential', mode='sequential')
        self.masks, self.out_mask = create_masks(self.degrees)
        self.mlp = MaskedMLP(self.n_hiddens, self.act_fn, self.masks)
        self.output_layer = OutputLayer(self.in_dim, self.out_mask)

    
    def __call__(self, inputs):
        h = self.mlp(inputs) 
        m, logp = self.output_layer(h) 
        return m, logp


    def gen(self, key, shape, u=None):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """
        x = jnp.zeros(shape)
        u = random.normal(key, shape) if u is None else u

        for i in range(1, shape[1] + 1):
            h = self.mlp(x)
            m, logp = self.output_layer(h) 
            idx = jnp.argwhere(self.degrees[0] == i)[0, 0]
            x = x.at[:, idx].set(m[:, idx] + jnp.exp(jnp.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx])
        return x
