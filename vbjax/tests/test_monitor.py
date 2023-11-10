import numpy
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

    fmri = b_sample(buf)
    assert fmri.shape == (n,)

