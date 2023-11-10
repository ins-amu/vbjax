import jax.numpy as np
from .loops import heun_step
from .neural_mass import BOLDTheta, bold_dfun


def make_timeavg(shape):
    "Make a time average monitor."
    new = lambda : {'y': np.zeros(shape), 'n': 0}
    def step(buf, x):
        return {'y': buf['y'] + x,
                'n': buf['n'] + 1}
    def sample(buf):
        return new(), buf['y'] / buf['n']
    return new(), step, sample


def compute_sarvas_gain(q, r, o, att, Ds=0, Dc=0) -> np.ndarray:
    # https://gist.github.com/maedoc/add7c3206f81d59105753a04f7c1fcf4
    pass


def make_gain(gain, shape):
    "Make a gain-matrix monitor suitable for sEEG, EEG & MEG."
    tavg_shape = gain.shape[:1] + shape[1:]
    buf, tavg_step, tavg_sample = make_timeavg(tavg_shape)
    step = lambda b, x: tavg_step(b, gain @ x)
    return buf, step, tavg_sample


def make_bold(shape, dt, p: BOLDTheta):
    "Make a BOLD fMRI monitor."
    sfvq = np.ones((4,) + shape)
    sfvq = sfvq.at[0].set(0)
    def step(sfvq, x):
        return heun_step(sfvq, bold_dfun, dt, x, p)
    def sample(buf):
        s, f, v, q = buf
        return p.v0 * (p.k1*(1 - q) + p.k2*(1 - q / v) + p.k3*(1 - v))
    return sfvq, step, sample


def make_fc(shape, period):
    # welford online cov estimate yields o(1) backprop memory usage
    # https://github.com/maedoc/tvb-fut/blob/master/lib/github.com/maedoc/tvb-fut/stats.fut#L9
    pass

def make_fft(shape, period):
    # incremental ft doesn't really exist, do windowed instead
    pass

# TODO sliding window versions of those

# monitors don't have same periods:
# core loop function takes one step
# does that enough to produce the largest period?
# @jax.checkpoint to lower memory usage?
