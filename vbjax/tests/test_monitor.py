import numpy
import jax
import jax.numpy as np
import vbjax as vb
import pytest

def test_timeavg():
    buf, ta_step, ta_sample = vb.make_timeavg((4,))

    buf = ta_step(buf, 1.0)
    buf = ta_step(buf, np.r_[:4])
    buf, ta = ta_sample(buf)
    numpy.testing.assert_allclose(ta, (1+np.r_[:4])/2)

    buf = ta_step(buf, 2.0)
    buf, ta = ta_sample(buf)
    numpy.testing.assert_allclose(ta, 2.0)


def test_offline():
    buf, ta_step, ta_sample = vb.make_timeavg((1,))
    bufo, _, _ = vb.make_timeavg((1,))
    ta_off = vb.make_offline(ta_step, ta_sample)
    xs = vb.randn(20, 100)
    for j, x_j in enumerate(xs):
        for i, x_ji in enumerate(x_j):
            buf = ta_step(buf, x_ji)
        buf, ta = ta_sample(buf)
        bufo, tao = ta_off(bufo, x_j)
        numpy.testing.assert_allclose(tao, ta)

def test_gain():
    gain = vb.randn(4, 4)
    ones = np.ones((4, ))
    buf, g_step, g_sample = vb.make_gain(gain, ones.shape)

    buf = g_step(buf, ones)
    buf = g_step(buf, ones)
    buf = g_step(buf, ones)

    buf, eeg = g_sample(buf)

    numpy.testing.assert_allclose(eeg, gain.sum(axis=1))


def test_cov_fc():
    xs = vb.randn(1000, 16)*2 + 1.0
    fc, fc_step, fc_sample = vb.make_fc(xs[0].shape)
    fcol = vb.make_offline(fc_step, fc_sample)
    _, s_fc = fcol(fc, xs)
    numpy.testing.assert_allclose(s_fc, np.identity(16), 0.09, 0.09)


def test_bold():
    n = 8
    dt = 0.1
    p = vb.bold_default_theta
    buf, b_step, b_sample = vb.make_bold((n, ), dt, p)

    buf = b_step(buf, vb.randn(n))
    buf = b_step(buf, vb.randn(n))
    buf = b_step(buf, vb.randn(n))

    _, fmri = b_sample(buf)
    assert fmri.shape == (n,)


def setup_multiple_periods(unroll, checkpoint):

    # setup monitors
    eeg_gain = vb.randn(64, 32)
    eeg_buf, eeg_step, eeg_sample = vb.make_gain(eeg_gain)
    eeg2_buf, eeg2_step, eeg2_sample = vb.make_gain(eeg_gain)
    eeg2_offline_sample = vb.make_offline(eeg2_step, eeg2_sample)
    bold_buf, bold_step, bold_sample = vb.make_bold((eeg_gain.shape[1], ),
                                                0.1, vb.bold_default_theta)

    # our simulation state
    # TODO may be easier with jax_dataclasses
    sim = {
        'eeg_buf': eeg_buf,
        'eeg2_buf': eeg2_buf,
        'bold_buf': bold_buf,
        'freq': 0.1,
    }

    # inner scan steps neural dynamics & monitor states
    @(jax.checkpoint if checkpoint else lambda f: f)
    def op1(sim, t):
        key_t = jax.random.PRNGKey(t)
        # insert neural dynamics here
        x = jax.random.normal(key_t, shape=(eeg_gain.shape[1],))
        # update monitors
        sim['eeg_buf'] = eeg_step(sim['eeg_buf'], np.sin(x * sim['freq']))
        sim['bold_buf'] = bold_step(sim['bold_buf'], np.sin(x) * 0.25 + 1.0)
        return sim, x

    # next scan samples eeg monitors
    def op2(sim, t_):
        # sample eeg w/ period of 10*dt
        sim, raw = jax.lax.scan(op1, sim, t_ * 10 + np.r_[:10],
                                unroll=10)
        sim['eeg_buf'], eeg_t = eeg_sample(sim['eeg_buf'])
        sim['eeg2_buf'], eeg2_t = eeg2_offline_sample(sim['eeg2_buf'],
                                                      np.sin(raw * sim['freq']))
        return sim, (raw, eeg_t, eeg2_t)

    # outer scan steps from one bold sample to the next
    def op3(sim, T):
        # run for 5 samples of eeg
        sim, (raw, eeg, eeg2) = jax.lax.scan(op2, sim, T*50 + np.r_[:5],
                                       unroll=5 if unroll else 1)
        # sample fmri w/ period of 5*10*dt
        _, fmri = bold_sample(sim['bold_buf'])
        return sim, (raw, eeg, eeg2, fmri)

    return sim, op3

def test_multiple_periods():
    sim, op3 = setup_multiple_periods(False, False,)
    ts = np.r_[:10]
    sim, (raw, eeg, eeg2, fmri) = jax.lax.scan(op3, sim, ts)
    assert raw.shape == (ts.size, 5, 10, 32)
    assert eeg.shape == (ts.size, 5, 64)
    assert eeg2.shape == (ts.size, 5, 64)
    assert fmri.shape == (ts.size, 32)
    numpy.testing.assert_allclose(eeg2, eeg, 2e-3, 1e-4)


@pytest.mark.parametrize('opts', [
    f'{args} {dev}'
    for args in ['jit grad','jit grad ckp','jit','']
    for dev in 'cpu,gpu'.split(',')])
def test_multiple_periods_perf(benchmark, opts):
    opts = opts.split(' ')
    device_name = 'cpu' if 'cpu' in opts else 'gpu'
    try:
        device = jax.devices(device_name)[0]
    except RuntimeError: # no gpu
        return
    unroll = 'unroll' in opts
    with jax.default_device(device):
        sim, op3 = setup_multiple_periods(unroll, 'ckp' in opts)
        ts = np.r_[:100]
        def run(freq, sim):
            sim = sim.copy()
            sim['freq'] = freq
            sim, (raw, eeg, eeg2, fmri) = jax.lax.scan(
                op3, sim, ts,
                unroll=10 if unroll else 1)
            return np.sum(np.square(eeg))
        if 'grad' in opts:
            run = jax.grad(run)
            assert np.abs(run(0.2,sim)) > 0
        if 'jit' in opts:
            run = jax.jit(run)
            run(0.2, sim)
        benchmark(lambda : run(0.2, sim))
