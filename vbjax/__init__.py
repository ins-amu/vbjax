from .loops import make_sde, make_ode, make_dde, make_sdde
from .shtlc import make_shtdiff
from .neural_mass import (
        JRState, JRTheta, jr_dfun, jr_default_theta,
        MPRState, MPRTheta, mpr_dfun, mpr_default_theta,
        )
from .regmap import make_region_mapping
from .coupling import make_diff_cfun, make_linear_cfun
from .connectome import make_conn_latent_mvnorm
from .sparse import make_spmv
from .layers import make_dense_layers
from .diagnostics import shrinkage_zscore
from .embed import embed_neural_flow, embed_polynomial, embed_gradient, embed_autoregress
from ._version import __version__


def use_many_cores(n):
    import os
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=%d' % n


# some random setup for convenience
from jax import random
key = random.PRNGKey(42)
keys = random.split(key, 100)
def randn(*shape, key=key):
    return random.normal(key, shape)

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

