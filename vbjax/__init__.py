def _say(msg): # TODO use logging
    print(f'c[vbjax] {msg}')

_say('███▒▒▒▒▒▒▒ loading')

# default to setting up many cores
def _use_many_cores():
    import os, sys
    import multiprocessing as mp
    if 'XLA_FLAGS' not in os.environ:
        n = mp.cpu_count()
        value = '--xla_force_host_platform_device_count=%d' % n
        os.environ['XLA_FLAGS'] = value
        _say(f'(ﾉ☉ヮ⚆)ﾉ ⌒*:･ﾟ✧ can haz {n} cores')
        return n
    else:
        _say('XLA_FLAGS already set\n')
        return 1

cores = _use_many_cores()

# import stuff
from .custom_loops import make_sde, make_ode, make_dde, make_sdde, heun_step, make_continuation
from .noise_generator import make_noise_generator, spectral_exponent
from .shtlc import make_shtdiff
from .neural_mass import (
    JRState, JRTheta, jr_dfun, jr_default_theta,
    MPRState, MPRTheta, mpr_dfun, mpr_default_theta, mpr_r_positive,
    BOLDTheta, compute_bold_theta, bold_default_theta, bold_dfun,
    BVEPTheta, bvep_default_theta, bvep_dfun, DCMTheta, dcm_dfun,
    DopaTheta, dopa_dfun, dopa_default_theta, dopa_default_initial_state,
    dopa_net_dfun, dopa_r_positive, dopa_gfun_add, dopa_gfun_mulr,
        )
from .regmap import make_region_mapping
from .coupling import (
        make_diff_cfun, make_linear_cfun, make_delay_helper, delay_apply,
        )
from .connectome import make_conn_latent_mvnorm
from .sparse import make_spmv, csr_to_jax_bcoo, make_sg_spmv
from .monitor import (
    make_timeavg, make_bold, make_gain, make_offline, make_cov, make_fc)
from .layers import (make_dense_layers, create_degrees, create_masks, 
    MaskedLayer, MaskedMLP, OutputLayer)
from .ml_models import GaussianMADE, MAF
from .diagnostics import shrinkage_zscore
from .embed import embed_neural_flow, embed_polynomial, embed_gradient, embed_autoregress
from .util import to_jax, to_np, tuple_meshgrid, tuple_ravel, tuple_shard
from .train_utils import (eval_model, train_step, log_likelihood_MADE, 
    log_likelihood_MAF, grad_func)

from ._version import __version__

# some random setup for convenience
import jax
platform = jax.local_devices()[0].platform
is_gpu = platform == 'gpu'
is_cpu = platform == 'cpu'

from jax import random
from jax import numpy as np
key = random.PRNGKey(42)
keys = random.split(key, 100)
def randn(*shape, key=key):
    return random.normal(key, shape)

def rand(*shape, key=key):
    return random.uniform(key, shape)

def randn_connectome(n, key=key):
    w, l = np.abs(randn(2, n, n, key=key))
    l = l * 100
    return w+w.T, l+l.T

# some simple plots
def plot_states(xs, names=None, jpg=None, show=False):
    import pylab as pl
    names = names or ['x%d' % i for i in range(xs.shape[1])]
    for i in range(xs.shape[1]):
        pl.subplot(xs.shape[1], 1, i+1)
        pl.plot(xs[:, i], 'k', alpha=0.3)
        pl.ylabel(names[i])
        pl.xlabel('time')
        pl.grid(1)
    pl.tight_layout()
    if jpg:
        pl.savefig(jpg if '.jpg' in jpg else jpg + '.jpg')
    if show:
        pl.show()

def make_field_gif(xt, gifname, fps=15):
    import pylab as pl
    import matplotlib.animation as animation
    fig, ax = pl.subplots()
    fig.set_size_inches((4,2))
    im = ax.imshow(xt[0,0])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Field activity')
    fig.tight_layout()
    def animate(i):
        im.set_data(xt[i+1,0])
        im.set_clim()
        return im,
    ani = animation.FuncAnimation(fig, animate,
            repeat=True, frames=len(xt)-1, interval=20)
    writer = animation.PillowWriter(fps=fps, bitrate=400)
    ani.save(gifname, writer=writer)

_say('ᕕ(ᐛ)ᕗ ready')
