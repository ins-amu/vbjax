import jax
import jax.numpy as np
from .loops import heun_step
from .neural_mass import BOLDTheta, bold_dfun


def make_offline(step_fn, sample_fn, *args):
    "Compute monitor samples in an offline or batch fashion."
    def op(mon, x):
        mon = step_fn(mon, x)
        return mon, None
    def offline_sample(mon, xs):
        mon, _ = jax.lax.scan(op, mon, xs)
        mon, samp = sample_fn(mon)
        return mon, samp
    return offline_sample

# NB shape here is the input shape of neural activity

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


def make_gain(gain, shape=None):
    "Make a gain-matrix monitor suitable for sEEG, EEG & MEG."
    tavg_shape = gain.shape[:1] + (shape[1:] if shape else ())
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
        return buf, p.v0 * (p.k1*(1 - q) + p.k2*(1 - q / v) + p.k3*(1 - v))
    return sfvq, step, sample


def make_cov(shape):
    # welford online cov estimate yields o(1) backprop memory usage
    # https://github.com/maedoc/tvb-fut/blob/master/lib/github.com/maedoc/tvb-fut/stats.fut#L9
    assert len(shape) == 1, "FC over 1-D vectors for now."
    cov_shape = shape * 2
    def new():
        buf = {'count': 0,
               'mean': np.zeros(shape),
               'cov': np.zeros(cov_shape)
               }
        return buf
    def step(buf, x):
        count, mean, cov = buf['count'], buf['mean'], buf['cov']
        count = count + 1
        dx = x - mean
        mean = mean + dx / count
        cov = cov + np.outer(dx, dx)
        # count = count + 1
        # delta = val - mean
        # mean = mean + delta / count
        # delta2 = val - mean
        # m2 = m2 + delta * delta2
        return {'count': count,
                'mean': mean,
                'cov': cov}
    def sample(buf):
        var = buf['cov'] / buf['count']
        # std = np.sqrt(var)
        return new(), var
    return new(), step, sample


def make_fc(shape):
    new_buf, step, cov_sample = make_cov(shape)
    def fc_sample(buf):
        buf, cov = cov_sample(buf)
        # Rij = Cij / sqrt(Cii Cjj)
        cii = np.diag(cov)
        return new_buf.copy(), cov / np.sqrt(np.outer(cii, cii))
    return new_buf.copy(), step, fc_sample


def make_fft(shape, period):
    pass

# TODO sliding window versions of those

# @jax.checkpoint to lower memory usage?
