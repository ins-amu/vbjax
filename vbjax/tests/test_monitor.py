import numpy
import jax
import jax.numpy as np
import vbjax as vb


def test_timeavg():
    buf, ta_step, ta_sample = vb.make_timeavg((4,))

    buf = ta_step(buf, 1.0)
    buf = ta_step(buf, np.r_[:4])
    buf, ta = ta_sample(buf)
    numpy.testing.assert_allclose(ta, (1+np.r_[:4])/2)

    buf = ta_step(buf, 2.0)
    buf, ta = ta_sample(buf)
    numpy.testing.assert_allclose(ta, 2.0)


def test_gain():
    gain = vb.randn(4, 4)
    ones = np.ones((4, ))
    buf, g_step, g_sample = vb.make_gain(gain, ones.shape)

    buf = g_step(buf, ones)
    buf = g_step(buf, ones)
    buf = g_step(buf, ones)

    buf, eeg = g_sample(buf)

    numpy.testing.assert_allclose(eeg, gain.sum(axis=1))


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


def test_multiple_periods():

    # setup monitors
    eeg_gain = vb.randn(64, 32)
    eeg_buf, eeg_step, eeg_sample = vb.make_gain(eeg_gain)
    bold_buf, bold_step, bold_sample = vb.make_bold((eeg_gain.shape[1], ),
                                                0.1, vb.bold_default_theta)

    # our simulation state
    # TODO may be easier with jax_dataclasses
    sim = {
        'eeg_buf': eeg_buf,
        'bold_buf': bold_buf
    }

    # inner scan steps neural dynamics & monitor states
    def op1(sim, t):
        key_t = jax.random.PRNGKey(t)
        # insert neural dynamics here
        x = jax.random.normal(key_t, shape=(eeg_gain.shape[1],))
        # update monitors
        sim['eeg_buf'] = eeg_step(sim['eeg_buf'], x)
        sim['bold_buf'] = bold_step(sim['bold_buf'], np.sin(x) * 0.25 + 1.0)
        return sim, x

    # next scan samples eeg monitors
    def op2(sim, t_):
        # sample eeg w/ period of 10*dt
        sim, raw = jax.lax.scan(op1, sim, t_ * 10 + np.r_[:10])
        sim['eeg_buf'], eeg_t = eeg_sample(sim['eeg_buf'])
        return sim, (raw, eeg_t)

    # outer scan steps from one bold sample to the next
    def op3(sim, T):
        # run for 5 samples of eeg
        sim, (raw, eeg) = jax.lax.scan(op2, sim, T*50 + np.r_[:5])
        # sample fmri w/ period of 5*10*dt
        _, fmri = bold_sample(sim['bold_buf'])
        return sim, (raw, eeg, fmri)

    ts = np.r_[:10]
    sim, (raw, eeg, fmri) = jax.lax.scan(op3, sim, ts)

    assert raw.shape == (ts.size, 5, 10, 32)
    assert eeg.shape == (ts.size, 5, 64)
    assert fmri.shape == (ts.size, 32)

    # that covers most use cases
    # metrics like fcd can be at done at fmri time scale

