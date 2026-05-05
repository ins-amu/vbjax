import numpy as np
import jax
import jax.numpy as jp
import pytest

import vbjax as vb
from vbjax.crosscoder import _denorm


def _fake_triu(ns=40, nn=16, seed=0):
    rng = np.random.default_rng(seed)
    mats = rng.normal(size=(ns, nn, nn)).astype('f4')
    mats = 0.5 * (mats + mats.transpose(0, 2, 1))
    i, j = np.triu_indices(nn, k=1)
    return mats[:, i, j], mats


def test_triu_to_mat_roundtrip():
    triu, mats = _fake_triu()
    nn = mats.shape[1]
    i, j = np.triu_indices(nn, k=1)
    np.testing.assert_allclose(np.asarray(vb.triu_to_mat(jp.asarray(triu)))[:, i, j], triu)
    np.testing.assert_allclose(vb.triu_to_mat_np(triu)[:, i, j], triu)

    single = vb.triu_to_mat(jp.asarray(triu[0]))
    assert single.shape == (nn, nn)
    np.testing.assert_allclose(single, single.T)


def test_add_view_normalizations():
    triu, _ = _fake_triu()
    cc = vb.CrossCoder(variational=False)
    cc.add_view(triu, 'SC', normalize='zscore')
    cc.add_view(np.abs(triu), 'FA', normalize='logit', nonneg=True)
    cc.add_view(triu, 'FC', normalize='center')

    np.testing.assert_allclose(float(jp.mean(cc.conns[0])), 0.0, atol=1e-4)
    assert 0.5 < float(jp.std(cc.conns[0])) < 2.0
    assert cc.norm_types == ['zscore', 'logit', 'center']
    assert cc.nonneg == [False, True, False]


def test_denorm_roundtrip():
    triu, _ = _fake_triu()
    cc = vb.CrossCoder(variational=False)
    cc.add_view(triu, 'A', normalize='zscore')
    cc.add_view(np.abs(triu) + 1e-3, 'B', normalize='logit', nonneg=True)
    for i in range(2):
        reco = _denorm(cc.conns[i], cc.norm_types[i], cc.means[i],
                       cc.stds[i], cc.scales[i], cc.nonneg[i])
        target = np.abs(triu) + 1e-3 if i == 1 else triu
        np.testing.assert_allclose(np.asarray(reco), target, atol=1e-3)


@pytest.mark.parametrize('variational', [False, True])
def test_train_deterministic_and_variational(variational):
    triu, _ = _fake_triu(ns=30, nn=10)
    cc = vb.CrossCoder(variational=variational, chunked_training=True)
    cc.add_view(triu, 'SC', normalize='center')
    cc.tts = 20
    trace, wbs, cr = cc.train(nlat=3, lr=5e-3, niter=100, mb=8,
                              beta_end=1e-4, anneal_steps=50)
    assert len(trace) > 1
    assert 0.0 <= cr <= 1.0
    assert cc.arch == [3]

    z = cc.encode(3, 'SC', sample=False)
    assert z.shape == (triu.shape[0], 3)
    recon = cc.decode(3, 'SC', z)
    assert recon.shape == triu.shape
    conn = cc.decode_conn(3, 'SC', z)
    assert conn.shape == (triu.shape[0], 10, 10)
    np.testing.assert_allclose(np.asarray(conn), np.asarray(conn).transpose(0, 2, 1), atol=1e-5)


def test_cross_prediction_matches_loss():
    triu_a, _ = _fake_triu(ns=20, nn=8, seed=1)
    triu_b, _ = _fake_triu(ns=20, nn=8, seed=2)
    cc = vb.CrossCoder(variational=False, chunked_training=False)
    cc.add_view(triu_a, 'A', normalize='center')
    cc.add_view(triu_b, 'B', normalize='center')
    cc.tts = 14
    cc.train(nlat=2, lr=5e-3, niter=50, mb=8)

    loss_fn, _ = cc.make_loss()
    l = float(loss_fn(cc.wbs[0], cc.conns))
    assert np.isfinite(l)


def test_calc_mvn_and_decompose():
    triu, _ = _fake_triu(ns=30, nn=8)
    cc = vb.CrossCoder(variational=True, chunked_training=True)
    cc.add_view(triu, 'SC', normalize='zscore')
    cc.tts = 20
    cc.train(nlat=3, lr=5e-3, niter=80, mb=8,
             beta_end=1e-4, anneal_steps=40)
    mvn = cc.calc_mvn(3)
    assert isinstance(mvn, vb.MvNorm)
    assert mvn.mean.shape == (3,) and mvn.cov.shape == (3, 3)
    samp = mvn.sample(5)
    assert samp.shape == (5, 3)

    d = cc.decompose_latent(3)
    assert d['components'].shape[0] == 3
    assert d['explained_variance'].size == 3


def test_pickle_roundtrip(tmp_path):
    triu, _ = _fake_triu(ns=20, nn=8)
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu, 'SC', normalize='center')
    cc.tts = 14
    cc.train(nlat=2, lr=5e-3, niter=30, mb=8)

    fn = tmp_path / 'cc.pkl'
    cc.to_pkl(str(fn))
    cc2 = vb.CrossCoder.from_pkl(str(fn))
    assert cc2.parcs == cc.parcs
    assert cc2.arch == cc.arch
    z = cc.encode(2, 'SC')
    z2 = cc2.encode(2, 'SC')
    np.testing.assert_allclose(np.asarray(z), np.asarray(z2), atol=1e-5)


def test_from_numpy_array():
    _, mats = _fake_triu(ns=30, nn=12)
    cc = vb.CrossCoder.from_numpy_array(mats, tts=20, variational=False)
    assert cc.tts == 20
    assert cc.nonneg == [True]
    assert len(cc.parcs) == 1
    cc.train(nlat=2, lr=5e-3, niter=30, mb=8)
    assert cc.arch == [2]


def test_combine():
    triu_a, _ = _fake_triu(ns=10, nn=8, seed=3)
    triu_b, _ = _fake_triu(ns=10, nn=8, seed=4)
    cc1 = vb.CrossCoder(variational=False)
    cc1.add_view(triu_a, 'SC', normalize='center')
    cc2 = vb.CrossCoder(variational=False)
    cc2.add_view(triu_b, 'SC', normalize='center')
    cc = vb.CrossCoder.combine(cc1, cc2, shuffle=False)
    assert cc.conns[0].shape[0] == 20
    assert cc.tts == 10


def test_sweep_crosscoder():
    triu, _ = _fake_triu(ns=20, nn=8)
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu, 'SC', normalize='center')
    cc.tts = 14
    results, best = vb.sweep_crosscoder(
        cc, dims=[2], n_trials=2,
        lr_range=(1e-3, 5e-3), niter_range=(20, 30), mb_choices=(8,))
    assert len(results) == 2
    assert best is results[0]
    assert cc.wbs == []


def test_multi_view_cross_reconstruction():
    triu_a, _ = _fake_triu(ns=30, nn=10, seed=1)
    triu_b, _ = _fake_triu(ns=30, nn=10, seed=2)
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu_a, 'A', normalize='center')
    cc.add_view(triu_b, 'B', normalize='center')
    cc.tts = 20
    trace, wbs, cr = cc.train(nlat=3, lr=5e-3, niter=80, mb=8)
    assert len(trace) > 1
    assert 0.0 <= cr <= 1.0

    # Encode one view and decode the other (cross-reconstruction)
    z_a = cc.encode(3, 'A')
    rec_b = cc.decode(3, 'B', z_a)
    assert rec_b.shape == triu_b.shape

    # Overall loss should be finite with two views
    loss_fn, _ = cc.make_loss()
    l = float(loss_fn(cc.wbs[0], cc.conns))
    assert np.isfinite(l)


@pytest.mark.parametrize('variational', [False, True])
def test_train_not_chunked(variational):
    triu, _ = _fake_triu(ns=20, nn=8)
    cc = vb.CrossCoder(variational=variational, chunked_training=False)
    cc.add_view(triu, 'SC', normalize='center')
    cc.tts = 14
    trace, wbs, cr = cc.train(
        nlat=2, lr=5e-3, niter=20, mb=8,
        beta_end=1e-4 if variational else 0.0,
        anneal_steps=10 if variational else 0)
    assert len(trace) == 21
    assert 0.0 <= cr <= 1.0


def test_encode_sample_stochastic():
    triu, _ = _fake_triu(ns=20, nn=8)
    cc = vb.CrossCoder(variational=True, chunked_training=True)
    cc.add_view(triu, 'SC', normalize='center')
    cc.tts = 14
    cc.train(nlat=2, lr=5e-3, niter=20, mb=8,
             beta_end=1e-4, anneal_steps=10)
    mu = cc.encode(2, 'SC', sample=False)
    z1 = cc.encode(2, 'SC', sample=True, key=jax.random.PRNGKey(1))
    z2 = cc.encode(2, 'SC', sample=True, key=jax.random.PRNGKey(2))
    assert z1.shape == mu.shape == (triu.shape[0], 2)
    assert not np.allclose(np.asarray(z1), np.asarray(mu))
    assert not np.allclose(np.asarray(z1), np.asarray(z2))


def test_add_view_zero_std():
    triu = np.ones((20, 36), dtype='f4')
    cc = vb.CrossCoder(variational=False)
    cc.add_view(triu, 'Flat', normalize='zscore')
    # Constant data -> centered is 0 -> norm is 0
    np.testing.assert_allclose(np.asarray(cc.conns[0]), 0.0, atol=1e-6)
    cc.tts = 14
    cc.train(nlat=2, lr=5e-3, niter=10, mb=8)
    assert cc.arch == [2]


def test_multi_view_different_sizes():
    """Deterministic cross-reconstruction with different nn across views."""
    triu_a, _ = _fake_triu(ns=30, nn=10, seed=1)   # triu dim 45
    triu_b, _ = _fake_triu(ns=30, nn=16, seed=2)   # triu dim 120
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu_a, 'SC-079', normalize='center')
    cc.add_view(triu_b, 'SC-200', normalize='center')
    cc.tts = 20
    trace, wbs, cr = cc.train(nlat=4, lr=5e-3, niter=100, mb=8)
    assert len(trace) > 1
    assert 0.0 <= cr <= 1.0

    # Encode coarse -> decode to fine
    z = cc.encode(4, 'SC-079')
    assert z.shape == (30, 4)
    rec_fine = cc.decode(4, 'SC-200', z)
    assert rec_fine.shape == triu_b.shape

    # Encode fine -> decode to coarse
    z2 = cc.encode(4, 'SC-200')
    assert z2.shape == (30, 4)
    rec_coarse = cc.decode(4, 'SC-079', z2)
    assert rec_coarse.shape == triu_a.shape

    # Full connectome shapes
    conn_a = cc.decode_conn(4, 'SC-079', z2)
    assert conn_a.shape == (30, 10, 10)
    conn_b = cc.decode_conn(4, 'SC-200', z)
    assert conn_b.shape == (30, 16, 16)
    # Symmetry
    np.testing.assert_allclose(
        np.asarray(conn_a), np.asarray(conn_a).transpose(0, 2, 1), atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(conn_b), np.asarray(conn_b).transpose(0, 2, 1), atol=1e-5)

    # Confusion rate in [0, 1]
    cr2 = cc.calc_confusion_rate(4)
    assert 0.0 <= cr2 <= 1.0


def test_multi_view_different_sizes_variational():
    """Variational mode with heterogeneous parcellations."""
    triu_a, _ = _fake_triu(ns=30, nn=10, seed=1)
    triu_b, _ = _fake_triu(ns=30, nn=16, seed=2)
    cc = vb.CrossCoder(variational=True, chunked_training=True)
    cc.add_view(triu_a, 'SC-079', normalize='center')
    cc.add_view(triu_b, 'SC-200', normalize='center')
    cc.tts = 20
    cc.train(nlat=4, lr=5e-3, niter=100, mb=8,
             beta_end=1e-4, anneal_steps=50)

    # Encode with sampling returns different results each call
    z1 = cc.encode(4, 'SC-079', sample=True, key=jax.random.PRNGKey(1))
    z2 = cc.encode(4, 'SC-079', sample=True, key=jax.random.PRNGKey(2))
    assert z1.shape == (30, 4)
    assert not np.allclose(np.asarray(z1), np.asarray(z2))

    # Cross-decode shapes
    rec_fine = cc.decode(4, 'SC-200', z1)
    assert rec_fine.shape == triu_b.shape
    rec_coarse = cc.decode(4, 'SC-079', z2)
    assert rec_coarse.shape == triu_a.shape

    # MVN over heterogeneous views
    mvn = cc.calc_mvn(4)
    assert isinstance(mvn, vb.MvNorm)
    assert mvn.mean.shape == (4,)
    assert mvn.cov.shape == (4, 4)


def test_confusion_rate_different_sizes():
    """Confusion-rate matrix has correct shape with mixed nn."""
    triu_a, _ = _fake_triu(ns=30, nn=10, seed=5)
    triu_b, _ = _fake_triu(ns=30, nn=14, seed=6)
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu_a, 'A', normalize='center')
    cc.add_view(triu_b, 'B', normalize='center')
    cc.tts = 20
    cc.train(nlat=3, lr=5e-3, niter=80, mb=8)

    # Scalar confusion rate
    cr = cc.calc_confusion_rate(3)
    assert 0.0 <= cr <= 1.0

    # Raw confusion matrix from _conf_rates_det
    wbs = cc.wbs[cc.arch.index(3)]
    test_c = [c[20:] for c in cc.conns]
    crs = cc._conf_rates_det(wbs, test_c)
    assert crs.shape == (2, 2)
    assert np.all(crs >= 0) and np.all(crs <= 1)

def test_shuffle_roundtrip():
    triu_a, _ = _fake_triu(ns=30, nn=10, seed=10)
    triu_b, _ = _fake_triu(ns=30, nn=10, seed=11)
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu_a, 'SC', normalize='center')
    cc.add_view(triu_b, 'FC', normalize='center')
    cc.tts = 20
    cc.train(nlat=2, lr=5e-3, niter=30, mb=8)
    cr1 = cc.calc_confusion_rate(2)
    assert 0.0 <= cr1 <= 1.0

    perm = cc.shuffle(seed=99)
    assert isinstance(perm, np.ndarray)
    assert cc.conns[0].shape == (30, 45)
    assert cc.conns[1].shape == (30, 45)

    cc.train(nlat=2, lr=5e-3, niter=30, mb=8)
    cr2 = cc.calc_confusion_rate(2)
    assert 0.0 <= cr2 <= 1.0


def test_combine_variational():
    triu_a, _ = _fake_triu(ns=20, nn=8, seed=5)
    triu_b, _ = _fake_triu(ns=20, nn=8, seed=6)
    cc1 = vb.CrossCoder(variational=True, chunked_training=True)
    cc1.add_view(triu_a, 'SC', normalize='center')
    cc1.tts = 10
    cc2 = vb.CrossCoder(variational=True, chunked_training=True)
    cc2.add_view(triu_b, 'SC', normalize='center')
    cc2.tts = 10

    cc = vb.CrossCoder.combine(cc1, cc2, shuffle=False)
    assert cc.variational is True
    assert cc.conns[0].shape == (40, 28)
    assert cc.tts == 20

    cc.train(nlat=2, lr=5e-3, niter=30, mb=8,
             beta_end=1e-4, anneal_steps=15)
    assert cc.arch == [2]
    cr = cc.calc_confusion_rate(2)
    assert 0.0 <= cr <= 1.0


def test_pickle_roundtrip_variational(tmp_path):
    triu, _ = _fake_triu(ns=20, nn=8, seed=7)
    cc = vb.CrossCoder(variational=True, chunked_training=True)
    cc.add_view(triu, 'SC', normalize='zscore')
    cc.tts = 14
    cc.train(nlat=3, lr=5e-3, niter=50, mb=8,
             beta_end=1e-4, anneal_steps=25)

    fn = tmp_path / 'cc_var.pkl'
    cc.to_pkl(str(fn))
    cc2 = vb.CrossCoder.from_pkl(str(fn))
    assert cc2.variational is True
    assert cc2.parcs == cc.parcs
    assert cc2.arch == cc.arch

    z1 = cc.encode(3, 'SC', sample=False)
    z2 = cc2.encode(3, 'SC', sample=False)
    np.testing.assert_allclose(np.asarray(z1), np.asarray(z2), atol=1e-5)

    # Stochastic encode on loaded model should work and differ from mu
    z_stoch = cc2.encode(3, 'SC', sample=True, key=jax.random.PRNGKey(42))
    assert z_stoch.shape == z2.shape
    assert not np.allclose(np.asarray(z_stoch), np.asarray(z2))


def test_variational_confusion_sampling():
    """Variational confusion rate with posterior sampling vs point estimate."""
    triu, _ = _fake_triu(ns=30, nn=10)
    cc = vb.CrossCoder(variational=True, chunked_training=True)
    cc.add_view(triu, 'SC', normalize='center')
    cc.tts = 20
    cc.train(nlat=3, lr=5e-3, niter=50, mb=8,
             beta_end=1e-4, anneal_steps=25)
    nlat = 3

    cr_mean = cc.calc_confusion_rate(nlat, n_samples=0)
    cr_sampled = cc.calc_confusion_rate(nlat, n_samples=50)

    assert 0.0 <= cr_mean <= 1.0
    assert 0.0 <= cr_sampled <= 1.0


def test_confusion_matrix_api():
    """confusion_matrix returns full (n_views, n_views) matrix, deterministic."""
    triu_a, _ = _fake_triu(ns=30, nn=10, seed=1)
    triu_b, _ = _fake_triu(ns=30, nn=10, seed=2)
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu_a, 'A', normalize='center')
    cc.add_view(triu_b, 'B', normalize='center')
    cc.tts = 20
    cc.train(nlat=3, lr=5e-3, niter=80, mb=8)
    nlat = 3

    mat = cc.confusion_matrix(nlat)
    assert mat.shape == (2, 2)
    assert np.all(mat >= 0) and np.all(mat <= 1)

    # Diagonal mean should match self-recon confusion rate
    cr = cc.calc_confusion_rate(nlat)
    np.testing.assert_allclose(float(np.diag(mat).mean()), cr, atol=1e-6)


def test_confusion_matrix_variational():
    """confusion_matrix returns full matrix with posterior sampling."""
    triu_a, _ = _fake_triu(ns=30, nn=10, seed=1)
    triu_b, _ = _fake_triu(ns=30, nn=10, seed=2)
    cc = vb.CrossCoder(variational=True, chunked_training=True)
    cc.add_view(triu_a, 'A', normalize='center')
    cc.add_view(triu_b, 'B', normalize='center')
    cc.tts = 20
    cc.train(nlat=3, lr=5e-3, niter=50, mb=8,
             beta_end=1e-4, anneal_steps=25)
    nlat = 3

    mat = cc.confusion_matrix(nlat, n_samples=10)
    assert mat.shape == (2, 2)
    assert np.all(mat >= 0) and np.all(mat <= 1)




def test_encode_all():
    """encode_all returns dict of latent vectors for every view."""
    triu_a, _ = _fake_triu(ns=20, nn=8, seed=1)
    triu_b, _ = _fake_triu(ns=20, nn=10, seed=2)
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu_a, 'A', normalize='center')
    cc.add_view(triu_b, 'B', normalize='center')
    cc.tts = 14
    cc.train(nlat=3, lr=5e-3, niter=30, mb=8)

    all_z = cc.encode_all(3)
    assert set(all_z.keys()) == {'A', 'B'}
    assert all_z['A'].shape == (20, 3)
    assert all_z['B'].shape == (20, 3)

    # Should match individual encode calls
    np.testing.assert_allclose(np.asarray(all_z['A']),
                               np.asarray(cc.encode(3, 'A')), atol=1e-5)


def test_add_view_preserves_dtype():
    """add_view should not force-cast to float32."""
    if jax.config.jax_enable_x64:
        triu = np.random.randn(20, 28).astype(np.float64)
        cc = vb.CrossCoder(variational=False)
        cc.add_view(triu, 'SC', normalize='center')
        assert cc.conns[0].dtype == np.float64
    else:
        pytest.skip("float64 dtype preservation requires JAX x64 to be enabled in the test environment")
    triu32 = np.random.randn(20, 28).astype(np.float32)
    cc2 = vb.CrossCoder(variational=False)
    cc2.add_view(triu32, 'SC', normalize='center')
    assert cc2.conns[0].dtype == np.float32


def test_from_pkl_legacy_apvbt(tmp_path):
    """Load old apvbt-format pickle (no stds, scales, norm_types, nonneg)."""
    import pickle as _pickle
    rng = np.random.default_rng(42)
    ns, nn = 20, 8
    conn = rng.normal(size=(ns, nn * (nn - 1) // 2)).astype('f4')
    mean = conn.mean(axis=0)
    old_data = {
        'conns': [conn - mean],
        'means': [mean],
        'parcs': ['08-Schaefer'],
        'tts': 14,
        'wbs': [[
            ((rng.normal(size=(nn * (nn - 1) // 2, 2)).astype('f4'),
              rng.normal(size=2).astype('f4')),
             (rng.normal(size=(2, nn * (nn - 1) // 2)).astype('f4'),
              rng.normal(size=nn * (nn - 1) // 2).astype('f4')))
        ]],
    }
    fn = tmp_path / 'old.pkl'
    with open(str(fn), 'wb') as f:
        _pickle.dump(old_data, f)

    cc = vb.CrossCoder.from_pkl(str(fn))
    assert cc.parcs == ['08-Schaefer']
    assert cc.tts == 14
    assert cc.variational is False
    assert len(cc.stds) == 1
    assert cc.stds[0] == 1.0
    assert cc.norm_types == ['center']
    assert cc.nonneg == [False]
    # Should be able to encode with loaded weights
    z = cc.encode(2, '08-Schaefer')
    assert z.shape == (20, 2)  # all subjects when tts not passed
    z_test = cc.encode(2, '08-Schaefer', tts=14)
    assert z_test.shape == (6, 2)  # 20 - 14 test subjects



def test_sweep_keep_best():
    """sweep_crosscoder with keep_best=True retains best trial's weights."""
    triu, _ = _fake_triu(ns=20, nn=8)
    cc = vb.CrossCoder(variational=False, chunked_training=True)
    cc.add_view(triu, 'SC', normalize='center')
    cc.tts = 14
    results, best = vb.sweep_crosscoder(
        cc, dims=[2], n_trials=3,
        lr_range=(1e-3, 5e-3), niter_range=(20, 30), mb_choices=(8,),
        keep_best=True)
    assert len(results) == 3
    assert cc.wbs  # not empty — best weights kept
    assert len(cc.wbs) == 1
    assert cc.arch == [2]
    # The kept weights should produce a valid confusion rate
    cr = cc.calc_confusion_rate(2)
    assert 0.0 <= cr <= 1.0


