import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence, Optional
from jax._src.prng import PRNGKeyArrayImpl
import jax.random as random
from vbjax.layers import MaskedMLP, OutputLayer, create_degrees, create_masks
import jax
from flax.linen.initializers import zeros

class GaussianMADE(nn.Module):
    key: PRNGKeyArrayImpl
    in_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    input_order: str = 'sequential'
    mode: str = 'sequential'

    def setup(self):
        self.degrees = create_degrees(self.key, self.in_dim, self.n_hiddens, input_order=self.input_order, mode=self.mode)
        self.masks, self.out_mask = create_masks(self.degrees)
        self.mlp = MaskedMLP(self.n_hiddens, self.act_fn, self.masks)
        self.output_layer = OutputLayer(self.in_dim, self.out_mask)

    
    def __call__(self, inputs):
        h = self.mlp(inputs)
        m, logp = self.output_layer(h)
        return m, logp


    def gen(self, key, shape, u=None):
        x = jnp.zeros(shape)
        u = random.normal(key, shape) if u is None else u

        for i in range(1, shape[1] + 1):
            h = self.mlp(x)
            m, logp = self.output_layer(h) 
            idx = jnp.argwhere(self.degrees[0] == i)[0, 0]
            x = x.at[:, idx].set(m[:, idx] + jnp.exp(jnp.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx])
        return x


class MAF(nn.Module):
    key: PRNGKeyArrayImpl
    in_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    n_mades: int
    input_order: Optional[Sequence] = None
    mode: str = 'sequential'

    def setup(self, input_order: Optional[Sequence] = None):
        input_order = jnp.arange(1, self.in_dim+1) if input_order == None else input_order
        self.mades = [GaussianMADE(random.split(self.key), self.in_dim, self.n_hiddens, self.act_fn, input_order=input_order[::((-1)**(i%2))], mode=self.mode) for i in range(self.n_mades)]

    def __call__(self, inputs):
        u = inputs
        logdet_dudx = 0
        for made in self.mades:
            ms, logp = made(u)
            u = jnp.exp(0.5 * logp) * (u - ms)
            logdet_dudx += 0.5 * jnp.sum(logp, axis=1)
        return u, logdet_dudx

    def gen(self, key, shape, u=None):
        x = random.normal(key, shape) if u is None else u

        for made in self.mades[::-1]:
            x = made.gen(key, shape, x)
        return x



class Heun_step(nn.Module):
    dfun: Callable
    stvar: Optional[int] = 0
    external_i: Optional[int] = False
    adhoc: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, xs, p, i_ext):
        tmap = jax.tree_util.tree_map
        dt = 1.
        d1 = self.dfun((x, p), i_ext)
        xi = tmap(lambda x,d: x + dt*d, x, d1)
        # xi = tmap(lambda x,d,a: x + dt*d + a, x, d1, stimulus)

        d2 = self.dfun((xi, p), i_ext)
        nx = tmap(lambda x, d1,d2: x + dt*0.5*(d1 + d2), x, d1, d2)
        # nx = tmap(lambda x, d1,d2,a: x + dt*0.5*(d1 + d2) + a, x, d1, d2, stimulus)
        return nx, x


class Integrator(nn.Module):
    dfun: Callable
    step: Callable
    stvar: Optional[int] = 0
    adhoc: Optional[Callable] = None

    @nn.compact
    def __call__(self, c, xs, p=None, external_i=None):
        STEP = nn.scan(self.step,
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=(2, 2, 2),
                        out_axes=2
                        )
        return STEP(self.dfun, self.stvar, self.adhoc)(c, xs, p, external_i)


class MLP_Ode(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    step: Callable
    kernel_init: Callable = jax.nn.initializers.normal(1e-6)
    bias_init: Callable = jax.nn.initializers.normal(1e-6)
    integrate: Optional[bool] = True
    i_ext: Optional[bool] = True
    stvar: Optional[int] = 0
    p_mix: Optional[bool] = False

    def setup(self):
        self.p_layers = [nn.Dense(feat, kernel_init=self.kernel_init) for feat in self.n_hiddens[0]] if self.p_mix else None
        dims = self.n_hiddens[1:] if self.p_mix else self.n_hiddens
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.bias_init) for feat in dims]
        self.output = nn.Dense(self.out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)


    def fwd(self, x, i_ext):
        x, p = x
        if self.p_mix:
            for layer in self.p_layers[:-1]:
                p = layer(p)
                p = self.act_fn(p)
            p = self.p_layers[-1](p)
        x = jnp.c_[x, p, i_ext] if self.i_ext else x

        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        x = self.output(x)
        return x

    def prepare_stimulus(self, x, external_i, stvar):
        stimulus = jnp.zeros(x.shape)
        # stimulus = stimulus.at[:,stvar,:].set(external_i) if isinstance(external_i, jnp.ndarray) else stimulus
        return stimulus

    @nn.compact
    def __call__(self, inputs):
        if not self.integrate:
            (x, p), i_ext = inputs
            deriv = self.fwd((x, p), i_ext)
            return deriv

        (x, p), i_ext = inputs if self.i_ext else (inputs, None)

        integrate = Integrator(self.fwd, self.step)
        # initialize carry
        xs = jnp.zeros_like(x)
        # stimulus = self.prepare_stimulus(x, i_ext, self.stvar)
        x = x[...,0]
        traj = integrate(x, xs, p, i_ext)

        return traj[1]



class Simple_MLP(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    kernel_init: Callable = jax.random.normal
    extra_p: bool = False

    @nn.compact
    def __call__(self, x, i_ext):
        layers = [nn.Dense(feat, kernel_init=self.kernel_init*1e-6, bias_init=self.kernel_init*1e-6) for feat in self.n_hiddens]
        output = nn.Dense(self.out_dim, kernel_init=self.kernel_init*1e-6, bias_init=self.kernel_init*1e-6)
        x = jnp.c_[x[0], x[1]] if self.extra_p else x[0]
        for layer in layers:
            x = layer(x)
            x = self.act_fn(x)
        x = output(x)
        return x


class NeuralOdeWrapper(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    extra_p: int
    step: Optional[Callable] = Heun_step
    integrator: Optional[Callable] = Integrator
    network: Optional[Callable] = Simple_MLP
    kernel_init: Callable = jax.nn.initializers.normal(10e-3)
    integrate: Optional[bool] = True
    i_ext: Optional[bool] = True
    stvar: Optional[int] = 0

    @nn.compact
    def __call__(self, inputs):
        x, p, i_ext = inputs
        dfun = self.network(inputs, self.n_hiddens, self.act_fn, i_ext, extra_p=self.extra_p)
        if not self.integrate:
            deriv = dfun(inputs, i_ext)
            return deriv

        integrate = self.integrator(dfun, self.step)
        xs = jnp.zeros_like(x) # initialize carry
        # i_ext = self.prepare_stimulus(x, i_ext, self.stvar)
        x = x[...,0]
        return integrate(x, xs, p, i_ext)[1]



class Encoder(nn.Module):
    in_dim: int
    latent_dim: int
    act_fn: Callable
    n_hiddens: Sequence[int] = None

    def setup(self, n_hiddens: Optional[Sequence] = None):
        n_hiddens = n_hiddens[::-1] if n_hiddens else [self.in_dim, 4*self.latent_dim, 2*self.latent_dim, self.latent_dim][::-1]
        self.layers = [nn.Dense(feat) for feat in n_hiddens]

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        return x


class Decoder(nn.Module):
    in_dim: int
    latent_dim: int
    act_fn: Callable
    n_hiddens: Sequence[int] = None

    def setup(self, n_hiddens: Optional[Sequence] = None):
        n_hiddens = n_hiddens[::-1] if n_hiddens else [self.in_dim, 4*self.latent_dim, 2*self.latent_dim, self.latent_dim][::-1]
        self.layers = [nn.Dense(feat) for feat in n_hiddens]

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_fn(x)
        x = self.layers[-1](x)
        return x

class Autoencoder(nn.Module):
    latent_dim: int
    encoder_act_fn: Callable
    decoder_act_fn: Callable
    ode_act_fn: Callable
    ode: bool = False
    n_hiddens: Sequence[int] = None
    kernel_init: Callable = jax.nn.initializers.normal(10e-3)
    step: Optional[Callable] = Heun_step
    integrator: Optional[Callable] = Integrator
    network: Optional[Callable] = Simple_MLP
    i_ext: Optional[bool] = True
    ode_n_hiddens: Optional[Sequence] = None

    def integrate(self, encoded, L):
        xs =  jnp.ones((encoded.shape[0], encoded.shape[1], L)) # initialize carry
        dfun = self.network(encoded.shape[1], self.ode_n_hiddens, self.ode_act_fn)
        integrator = self.integrator(dfun, self.step)
        return integrator(encoded, xs)[1]

    @nn.compact
    def __call__(self, inputs):
        L = inputs.shape[-1]

        encoder = Encoder(inputs.shape[1], self.latent_dim, self.encoder_act_fn, self.n_hiddens)
        encoded = encoder(inputs[:,:,0]) if self.ode else encoder(inputs) # (N, )

        decoder = Decoder(inputs.shape[1], self.latent_dim, self.decoder_act_fn)
        y = decoder(encoded)
        if self.ode:
            y = self.integrate(encoded, L)

        return y

