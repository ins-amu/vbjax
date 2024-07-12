import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence, Optional
from collections import namedtuple, defaultdict
from jax._src.prng import PRNGKeyArrayImpl
import jax.random as random
from vbjax.layers import MaskedMLP, OutputLayer, create_degrees, create_masks
import jax
from flax.linen.initializers import zeros
import tqdm
from .neural_mass import BOLDTheta, bold_dfun

DelayHelper = namedtuple('DelayHelper', 'Wt lags ix_lag_from max_lag n_to n_from')

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
    dt: float = 1.0
    stvar: Optional[int] = 0
    external_i: Optional[int] = False
    adhoc: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, xs, p, i_ext):
        tmap = jax.tree_util.tree_map
        d1 = self.dfun((x, p), i_ext)
        xi = tmap(lambda x,d: x + self.dt*d, x, d1)
        # xi = tmap(lambda x,d,a: x + dt*d + a, x, d1, stimulus)

        d2 = self.dfun((xi, p), i_ext)
        nx = tmap(lambda x, d1,d2: x + self.dt*0.5*(d1 + d2), x, d1, d2)
        # nx = tmap(lambda x, d1,d2,a: x + dt*0.5*(d1 + d2) + a, x, d1, d2, stimulus)
        return nx, x


class Buffer_step(nn.Module):
    dfun: Callable
    adhoc: Callable
    nh: int
    dt: float = 1.0
    t_step: int = 0
    stvar: Optional[int] = 0
    external_i: Optional[int] = False
    

    @nn.compact
    def __call__(self, buf, dWt, t, p=None, i_ext=0):
        t = t[0][0].astype(int) # retrieve time step
        nh = self.nh
        
        tmap = jax.tree_util.tree_map
        x = tmap(lambda buf: buf[nh + t], buf)
        d1 = self.dfun(buf, x, nh + t)

        xi = tmap(lambda x,d,n: x + self.dt*d + n, x, d1, dWt)
        # xi = self.adhoc(xi)
        xi = tmap(self.adhoc, xi)

        d2 = self.dfun(buf, xi, nh + t + 1)
        nx = tmap(lambda x,d1,d2,n: x + self.dt*0.5*(d1 + d2) + n, x, d1, d2, dWt)
        nx = tmap(self.adhoc, nx)
        buf = tmap(lambda buf, nx: buf.at[nh + t + 1].set(nx), buf, nx)
        return buf, nx


class Integrator(nn.Module):
    dfun: Callable
    step: Callable
    adhoc: Callable
    dt: float = 1.0
    stvar: Optional[int] = 0
    nh: Optional[int] = None

    @nn.compact
    def __call__(self, c, xs, t_count, p=None, external_i=None, t=0):
        STEP = nn.scan(self.step,
                        variable_broadcast=["params", "noise"],
                        split_rngs={"params": False, "noise": True},
                        in_axes=(0, 0, 0, 0),
                        out_axes=0
                        )
        return STEP(self.dfun, self.adhoc, self.nh, self.dt, self.stvar)(c, xs, t_count, p, external_i)


class TVB(nn.Module):
    tvb_p: namedtuple
    dh: DelayHelper
    dfun: Callable
    nst_vars: int
    n_pars: int
    dfun_pars: defaultdict
    dt: float = 0.1
    seed: int = 42
    integrator: Optional[Callable] = Integrator
    step: Callable = Buffer_step
    adhoc: Callable = lambda x : x
    gfun: Callable = lambda x : x

    def delay_apply(self, dh: DelayHelper, t, buf):
        return (dh.Wt * buf[t - dh.lags, dh.ix_lag_from, :]).sum(axis=1)
    
    def fwd(self, nmm, region_pars):
        def tvb_dfun(buf, x, t):
            coupled_x = self.delay_apply(self.dh, t, buf[...,:self.nst_vars])
            coupling_term = coupled_x[:,:1] # firing rate coupling only for QIF

            # concat state, nmm_p (eta, p_synap), and i_ext (coupling_term) for MLP
            x = jnp.c_[x, region_pars, self.tvb_p['g']*coupling_term]
            return nmm(x)
        return tvb_dfun

    def noise_fill(self, buf, nh, key):
        # dWt =jax.random.normal(key, buf[nh+1:].shape)
        dWt = jax.random.normal(key, buf[nh+1:].transpose(0,2,1).shape)
        dWt = dWt.transpose(0,2,1)
        noise = self.gfun(dWt, jnp.sqrt(self.dt))
        buf = buf.at[nh+1:].set(noise)
        return buf

    def initialize_buffer(self, key):        
        dh = self.dh
        nh = int(dh.max_lag)
        buf = jnp.zeros((nh + int(1/self.dt) + 1, dh.n_from, self.nst_vars))

        initial_cond = jnp.c_[
            jax.random.uniform(key=key, shape=(dh.n_from, 1), minval=0.1, maxval=2.0),
            jax.random.uniform(key=key, shape=(dh.n_from, 1), minval=-2., maxval=1.5)
            ]

        # horizon is set at the start of the buffer because rolled at the start of chunk
        buf = buf.at[int(1/self.dt):,:,:self.nst_vars].add( initial_cond )
        return buf    

    def chunk(self, module, buf, key):
        nh = int(self.dh.max_lag)
        buf = jnp.roll(buf, -int(1/self.dt), axis=0)
        buf = self.noise_fill(buf, nh, key)
        dWt = buf[nh+1:] # initialize carry noise filled

        # pass time count to the scanned integrator
        t_count = jnp.tile(jnp.arange(int(1/self.dt))[...,None,None], (84, 2)) # (buf_len, regions, state_vars)
        return module(buf, dWt, t_count)



    # def sim_metrics(self, rv):


    @nn.compact
    def __call__(self, inputs,sim_len=30, eta=-5.,  batch_size = 8):
        seed, eta = inputs
        to_fit = self.param('to_fit',
                        lambda key, x: self.tvb_p['to_fit'], # Initialization function
                        self.tvb_p['to_fit'].shape)
        
        region_pars = jnp.c_[-5. * jnp.ones((self.dh.n_from, 1)), jnp.ones((self.dh.n_from, 1))]
        region_pars = region_pars.at[1,0].set(eta)
        key = jax.random.PRNGKey(seed)
        jax.debug.print("fit_num {x}", x=to_fit)
        buf = self.initialize_buffer(key)
        
        nmm = lambda x: self.dfun(self.dfun_pars, x)
        tvb_dfun = self.fwd(nmm, region_pars)
        # jax.debug.print("key {x}", x=self.adhoc)
        module = self.integrator(tvb_dfun, self.step, self.adhoc, self.dt, nh=int(self.dh.max_lag))
        run_chunk = nn.scan(self.chunk.__call__)
        run_sim = nn.scan(run_chunk)
        # sim_batch = nn.scan(run_sim)
        
        buf, rv = run_sim(module, buf, jax.random.split(key, (sim_len//2, 2000)))

        # if type(batch_size)==int:
        #     buf, rv = sim_batch(module, buf, jax.random.split(key, (batch_size, sim_len, 1000)))
        # else:
        #     buf, rv = sim_batch(module, buf, jax.random.split(key, (batch_size.astype(int), sim_len.astype(int), 1000)))
        # return rv.squeeze().reshape(batch_size, -1, self.dh.n_from, self.nst_vars)
        return rv.squeeze().reshape(-1, self.dh.n_from, self.nst_vars)



class MLP_Ode(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    step: Callable
    dt: float = 1.0
    additive: bool = False
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
        jax.debug.print("🤯 {x} 🤯", x=x.shape)
        jax.debug.print("🤯 {x} 🤯", x=i_ext.shape)
        if self.p_mix:
            for layer in self.p_layers[:-1]:
                p = layer(p)
                p = self.act_fn(p)
            p = self.p_layers[-1](p)
        if self.additive:
            x = jnp.c_[x, p] if self.i_ext else x
        else:
            x = jnp.c_[x, p, i_ext] if self.i_ext else x

        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        x = self.output(x)
        if self.additive:
            x = x.at[:,1:2].set(x[:,1:2] + i_ext)
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

        integrate = Integrator(self.fwd, self.step, self.dt)
        # initialize carry
        xs = jnp.zeros_like(x)
        # stimulus = self.prepare_stimulus(x, i_ext, self.stvar)
        x = x[...,0]
        jax.debug.print("🤯 x shape {x} 🤯", x=x.shape)
        jax.debug.print("🤯 xs shape {x} 🤯", x=xs.shape)
        jax.debug.print("🤯 iext shape {x} 🤯", x=i_ext.shape)
        traj = integrate(x, xs, p, i_ext)

        return traj[1]


class Simple_MLP(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    kernel_init: Callable = jax.nn.initializers.normal(1e-6)
    extra_p: bool = False
    coupled: bool = True

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.kernel_init) for feat in self.n_hiddens]
        self.output = nn.Dense(self.out_dim, kernel_init=self.kernel_init, bias_init=self.kernel_init)
    
    @nn.compact
    def __call__(self, x):
        # x = jnp.c_[x, i_ext] if self.coupled else x
        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        x = self.output(x)
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

