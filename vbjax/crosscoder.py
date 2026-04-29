"""
CrossCoder for amortized inference of whole-brain connectomes.

A single ``CrossCoder`` maps multiple views (e.g. different parcellations
or imaging modalities) of a connectome cohort into a shared low-dimensional
latent space via strictly linear encoders and decoders.  The variational
mode places a Gaussian over the latent and regularises towards a standard
normal, which helps when training data pools heterogeneous cohorts.
"""

import pickle
import functools
import numpy
import tqdm
import jax
import jax.numpy as np
from dataclasses import dataclass, field
from typing import Any, List

from vbjax import _optimizers as optimizers


SEED = 42
_LOGVAR_CLIP = (-15.0, 5.0)
_GRAD_CLIP = 5.0


@dataclass
class TrainedArch:
    """One trained cross-coder architecture."""
    nlat: int
    wbs: Any                          # list of (encoder, decoder) per view
    history: dict = field(default_factory=dict)
    variational: bool = False


def triu_to_mat(triu):
    "Fold flat upper-triangular vectors into symmetric square matrices."
    squeeze = triu.ndim == 1
    if squeeze:
        triu = triu[None]
    ns, nf = triu.shape
    nn = int(numpy.ceil((1 + numpy.sqrt(1 + 8 * nf)) / 2))
    i, j = np.triu_indices(nn, k=1)
    mat = np.zeros((ns, nn, nn), triu.dtype)
    mat = mat.at[:, i, j].set(triu).at[:, j, i].set(triu)
    return mat[0] if squeeze else mat


def triu_to_mat_np(triu):
    "NumPy equivalent of :func:`triu_to_mat` for post-training paths."
    squeeze = triu.ndim == 1
    if squeeze:
        triu = triu[None]
    ns, nf = triu.shape
    nn = int(numpy.ceil((1 + numpy.sqrt(1 + 8 * nf)) / 2))
    i, j = numpy.triu_indices(nn, k=1)
    mat = numpy.zeros((ns, nn, nn), triu.dtype)
    mat[:, i, j] = triu
    mat[:, j, i] = triu
    return mat[0] if squeeze else mat


def _xavier(key, shape, scale=1.0):
    return jax.random.normal(key, shape) * np.sqrt(scale / shape[0])


class MvNorm:
    "Multivariate normal with persistent PRNG key for sampling."

    def __init__(self, us, mean, cov, key=None):
        self.us = us
        self.mean = mean
        self.cov = cov
        self.key = key if key is not None else jax.random.PRNGKey(SEED)

    def sample(self, n):
        self.key, k = jax.random.split(self.key)
        return jax.random.multivariate_normal(k, self.mean, self.cov, shape=(n,))


def _denorm(flat, ntype, mean, std, scale, nonneg):
    "Invert the per-view normalization applied by :meth:`CrossCoder.add_view`."
    if ntype == 'logit':
        probs = jax.nn.sigmoid(flat * std + mean)
        eps = 1e-6
        out = (probs - eps) / (1 - 2 * eps) * scale
    else:
        out = flat * std + mean
    if nonneg:
        out = np.maximum(out, 0.0)
    return out


class CrossCoder:
    """
    Multi-view linear auto-encoder over flat upper-triangular connectomes.

    Parameters
    ----------
    variational : bool
        If True, the encoder emits a Gaussian over the latent rather than a
        point estimate, and training minimises ``MSE + β·KL``.
    chunked_training : bool
        If True, each ``train`` call compiles inner steps into a ``lax.scan``
        and returns to Python only for logging.  Disable for step-wise debug.
    """

    def __init__(self, variational=True, chunked_training=True):
        self.variational = variational
        self.chunked_training = chunked_training
        self.conns = []
        self.means = []
        self.stds = []
        self.scales = []
        self.parcs = []
        self.norm_types = []
        self.nonneg = []
        self.wbs = []
        self.tts = None
        self.history = []
        self.archs: List[TrainedArch] = []

    _pkl_keys = ('variational', 'chunked_training', 'conns', 'means', 'stds',
                 'scales', 'parcs', 'tts', 'wbs', 'history', 'norm_types', 'nonneg',
                 'archs')

    def to_pkl(self, fname):
        data = {k: getattr(self, k) for k in self._pkl_keys}
        # Convert TrainedArch dataclasses to dicts for pickling
        if 'archs' in data:
            data['archs'] = [{'nlat': a.nlat, 'wbs': a.wbs, 'history': a.history,
                              'variational': a.variational} for a in data['archs']]
        with open(fname, 'wb') as fd:
            pickle.dump(data, fd)

    @classmethod
    def from_pkl(cls, fname):
        with open(fname, 'rb') as fd:
            data = pickle.load(fd)
        self = cls(variational=data.get('variational', False),
                   chunked_training=data.get('chunked_training', True))
        for key in cls._pkl_keys[2:]:
            if key in data:
                setattr(self, key, data[key])
        self.conns = [np.asarray(c) for c in self.conns]
        self.means = [np.asarray(m) for m in self.means]
        self.stds = [float(s) for s in self.stds]
        if not self.norm_types:
            self.norm_types = ['center' if s == 1.0 else 'zscore' for s in self.stds]
        if not self.scales:
            self.scales = [1.0] * len(self.conns)
        if not self.nonneg:
            self.nonneg = [False] * len(self.conns)
        # Migration: rebuild archs from old-format pickles
        if not self.archs and self.wbs:
            for idx, wb in enumerate(self.wbs):
                hist = self.history[idx] if idx < len(self.history) else {}
                nlat = hist.get('nlat') if isinstance(hist, dict) else None
                if nlat is None:
                    # Infer nlat from weights
                    nlat = int(wb[0][0][0][1].size if self.variational else wb[0][0][1].size)
                self.archs.append(TrainedArch(nlat=nlat, wbs=wb, history=hist,
                                               variational=self.variational))
        elif self.archs and isinstance(self.archs[0], dict):
            # New-format pickle with dicts
            self.archs = [TrainedArch(**a) for a in self.archs]
        return self

    @classmethod
    def from_numpy_array(cls, weights, tts=None, parc='Schaefer-17Networks',
                         variational=False, chunked_training=True, normalize='center'):
        "Build a single-view CrossCoder from an ``(ns, nn, nn)`` connectome stack."
        weights = numpy.maximum(weights, 0.0)
        ns, nn, _ = weights.shape
        i, j = numpy.triu_indices(nn, k=1)
        self = cls(variational=variational, chunked_training=chunked_training)
        self.tts = tts if tts is not None else ns // 2
        self.add_view(weights[:, i, j], f'{nn}-{parc}', normalize=normalize, nonneg=True)
        return self

    def add_view(self, data, parc_name, normalize='zscore', nonneg=False):
        "Register a view, normalizing its flat upper-tri connectomes."
        data = np.asarray(data, dtype=np.float32)
        scale = 1.0
        std = 1.0

        if normalize == 'zscore':
            mean = np.mean(data, axis=0)
            centered = data - mean
            std = float(np.std(centered) + 1e-9)
            norm = centered / std
        elif normalize == 'center':
            mean = np.mean(data, axis=0)
            norm = data - mean
        elif normalize == 'logit':
            scale = float(np.max(data))
            eps = 1e-6
            x = (data / (scale + 1e-9)) * (1 - 2 * eps) + eps
            logits = np.log(x / (1 - x))
            mean = np.mean(logits, axis=0)
            std = float(np.std(logits) + 1e-9)
            norm = (logits - mean) / std
        else:
            mean = np.zeros(data.shape[1])
            norm = data

        self.conns.append(norm)
        self.means.append(mean)
        self.stds.append(std)
        self.scales.append(scale)
        self.parcs.append(parc_name)
        self.norm_types.append(normalize)
        self.nonneg.append(nonneg)

    def shuffle(self, seed=None):
        "Shuffle all views with a common permutation and return it."
        n = self.conns[0].shape[0]
        key = jax.random.PRNGKey(SEED if seed is None else seed)
        perm = jax.random.permutation(key, np.arange(n))
        self.conns = [c[perm] for c in self.conns]
        return numpy.asarray(perm)

    def make_wbs(self, nlat, key=None):
        "Initialize weights/biases for all views at given latent size."
        if key is None:
            key = jax.random.PRNGKey(SEED)
        wbs = []
        for c in self.conns:
            n = c.shape[1]
            if self.variational:
                key, k1, k2, k3 = jax.random.split(key, 4)
                w_mu = _xavier(k1, (n, nlat))
                w_lv = _xavier(k2, (n, nlat), scale=1e-4)
                w_dec = _xavier(k3, (nlat, n))
                enc = ((w_mu, np.zeros(nlat)), (w_lv, np.ones(nlat) * -10.0))
                dec = (w_dec, np.zeros(n))
            else:
                key, k1, k2 = jax.random.split(key, 3)
                enc = (_xavier(k1, (n, nlat), scale=1e-4), np.zeros(nlat))
                dec = (_xavier(k2, (nlat, n), scale=1e-4), np.zeros(n))
            wbs.append((enc, dec))
        return wbs


    def make_loss(self):
        "Build the cross-prediction loss and its gradient."
        if self.variational:
            @jax.jit
            def loss(wbs, conns, rng, beta):
                kl_tot = 0.0
                recon_tot = 0.0
                recon_list = []
                for i, (((w_mu, b_mu), (w_lv, b_lv)), _) in enumerate(wbs):
                    mu = conns[i] @ w_mu + b_mu
                    logvar = np.clip(conns[i] @ w_lv + b_lv, *_LOGVAR_CLIP)
                    kl = -0.5 * np.sum(1 + logvar - np.square(mu) - np.exp(logvar), axis=-1)
                    kl_tot += np.mean(kl)
                    rng, sub = jax.random.split(rng)
                    z = mu + np.exp(0.5 * logvar) * jax.random.normal(sub, mu.shape)
                    for j, (_, (w_dec, b_dec)) in enumerate(wbs):
                        mse = np.mean(np.square(z @ w_dec + b_dec - conns[j]))
                        recon_tot += mse
                        recon_list.append(mse)
                return recon_tot + beta * kl_tot, (recon_tot, kl_tot, recon_list)
            grad = jax.jit(jax.grad(lambda w, c, r, b: loss(w, c, r, b)[0]))
        else:
            @jax.jit
            def loss(wbs, conns):
                ll = 0.0
                for i, ((ew, eb), _) in enumerate(wbs):
                    u = conns[i] @ ew + eb
                    for j, (_, (dw, db)) in enumerate(wbs):
                        ll = ll + np.mean((u @ dw + db - conns[j]) ** 2)
                return ll
            grad = jax.jit(jax.grad(loss))
        return loss, grad


    def train(self, nlat, lr=3e-4, niter=2000, tts=None, mb=64,
              beta_start=0.0, beta_end=1e-3, anneal_steps=1500, key=None):
        """
        Fit a single architecture.  Appends the learned weights and trace
        into ``self.wbs`` / ``self.history`` and returns
        ``(trace, wbs, confusion_rate)``.
        """
        tts = tts or self.tts
        if tts is None:
            raise ValueError('set self.tts or pass tts= before training')

        train_c = [c[:tts] for c in self.conns]
        test_c = [c[tts:] for c in self.conns]
        wbs = self.make_wbs(nlat, key=key)
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(wbs)
        loss_fn, grad_fn = self.make_loss()
        mbkey = jax.random.PRNGKey(SEED)
        trace = []

        if self.variational:
            trace, wbs, log_freq = self._train_var(
                loss_fn, grad_fn, opt_update, get_params, opt_state, mbkey,
                train_c, test_c, tts, mb, niter,
                beta_start, beta_end, anneal_steps)
        else:
            trace, wbs, log_freq = self._train_det(
                loss_fn, grad_fn, opt_update, get_params, opt_state, mbkey,
                train_c, test_c, tts, mb, niter)

        self.wbs.append(wbs)
        hist = {'nlat': nlat, 'trace': trace,
                'log_freq': log_freq, 'variational': self.variational}
        self.history.append(hist)
        self.archs.append(TrainedArch(nlat=nlat, wbs=wbs, history=hist,
                                       variational=self.variational))
        cr = self.calc_confusion_rate(nlat, tts=tts)
        return trace, wbs, cr

    def _train_var(self, loss_fn, grad_fn, opt_update, get_params, opt_state,
                   mbkey, train_c, test_c, tts, mb, niter,
                   beta_start, beta_end, anneal_steps):
        def beta_at(it):
            return np.minimum(
                beta_start + (beta_end - beta_start) * (it / anneal_steps), beta_end)

        def step(i, opt_state, mbkey):
            beta = beta_at(i)
            mbkey, kb, kl = jax.random.split(mbkey, 3)
            idx = jax.random.randint(kb, (mb,), 0, tts)
            mb_c = [c[idx] for c in train_c]
            g = grad_fn(get_params(opt_state), mb_c, kl, beta)
            g = jax.tree.map(lambda x: np.clip(x, -_GRAD_CLIP, _GRAD_CLIP), g)
            return opt_update(i, g, opt_state), mbkey

        trace = []
        if self.chunked_training:
            log_freq = 50

            @functools.partial(jax.jit, static_argnums=(3,))
            def run_chunk(i0, opt_state, mbkey, n_steps):
                def body(carry, i):
                    opt_state, mbkey = carry
                    opt_state, mbkey = step(i, opt_state, mbkey)
                    return (opt_state, mbkey), None
                (opt_state, mbkey), _ = jax.lax.scan(
                    body, (opt_state, mbkey), np.arange(n_steps) + i0)
                return opt_state, mbkey

            pbar = tqdm.tqdm(total=niter + 1)
            i = 0
            while i <= niter:
                n = min(log_freq, niter + 1 - i)
                if n == 0:
                    break
                opt_state, mbkey = run_chunk(i, opt_state, mbkey, n)
                wbs = get_params(opt_state)
                beta = beta_at(i + n - 1)
                mbkey, kl = jax.random.split(mbkey)
                idx = jax.random.randint(kl, (mb,), 0, tts)
                log_mb = [c[idx] for c in train_c]
                l_tr, (r_tr, kl_tr, r_det) = loss_fn(wbs, log_mb, mbkey, beta)
                l_te, (r_te, kl_te, _) = loss_fn(wbs, test_c, mbkey, beta)
                trace.append([float(x) for x in [l_tr, l_te, r_tr, kl_tr] + r_det])
                pbar.set_description(f'R:{r_tr:.4f} KL:{kl_tr:.1f}')
                i += n
                pbar.update(n)
            pbar.close()
        else:
            log_freq = 1
            for i in (pbar := tqdm.trange(niter + 1)):
                opt_state, mbkey = step(i, opt_state, mbkey)
                wbs = get_params(opt_state)
                beta = beta_at(i)
                mbkey, kl = jax.random.split(mbkey)
                idx = jax.random.randint(kl, (mb,), 0, tts)
                log_mb = [c[idx] for c in train_c]
                l_tr, (r_tr, kl_tr, r_det) = loss_fn(wbs, log_mb, mbkey, beta)
                l_te, (r_te, kl_te, _) = loss_fn(wbs, test_c, mbkey, beta)
                trace.append([float(x) for x in [l_tr, l_te, r_tr, kl_tr] + r_det])
                pbar.set_description(f'R:{r_tr:.4f} KL:{kl_tr:.1f}')

        return trace, get_params(opt_state), log_freq

    def _train_det(self, loss_fn, grad_fn, opt_update, get_params, opt_state,
                   mbkey, train_c, test_c, tts, mb, niter):
        def step(i, opt_state, mbkey):
            mbkey, kb = jax.random.split(mbkey)
            idx = jax.random.randint(kb, (mb,), 0, tts)
            mb_c = [c[idx] for c in train_c]
            g = grad_fn(get_params(opt_state), mb_c)
            return opt_update(i, g, opt_state), mbkey

        trace = []
        if self.chunked_training:
            log_freq = 50

            @functools.partial(jax.jit, static_argnums=(3,))
            def run_chunk(i0, opt_state, mbkey, n_steps):
                def body(carry, i):
                    opt_state, mbkey = carry
                    opt_state, mbkey = step(i, opt_state, mbkey)
                    return (opt_state, mbkey), None
                (opt_state, mbkey), _ = jax.lax.scan(
                    body, (opt_state, mbkey), np.arange(n_steps) + i0)
                return opt_state, mbkey

            pbar = tqdm.tqdm(total=niter + 1)
            i = 0
            while i <= niter:
                n = min(log_freq, niter + 1 - i)
                if n == 0:
                    break
                opt_state, mbkey = run_chunk(i, opt_state, mbkey, n)
                wbs = get_params(opt_state)
                mbkey, kl = jax.random.split(mbkey)
                idx = jax.random.randint(kl, (mb,), 0, tts)
                log_mb = [c[idx] for c in train_c]
                ll_tr = float(np.log(loss_fn(wbs, log_mb)))
                ll_te = float(np.log(loss_fn(wbs, test_c)))
                trace.append((ll_tr, ll_te))
                ll0 = trace[0][1]
                pbar.set_description(f'-ll {ll0 - ll_te:.4f}')
                i += n
                pbar.update(n)
            pbar.close()
        else:
            log_freq = 1
            for i in (pbar := tqdm.trange(niter + 1)):
                opt_state, mbkey = step(i, opt_state, mbkey)
                wbs = get_params(opt_state)
                mbkey, _key = jax.random.split(mbkey)
                idx = jax.random.randint(_key, (mb,), 0, tts)
                log_mb = [c[idx] for c in train_c]
                ll_tr = float(np.log(loss_fn(wbs, log_mb)))
                ll_te = float(np.log(loss_fn(wbs, test_c)))
                trace.append((ll_tr, ll_te))
                ll0 = trace[0][1]
                pbar.set_description(f'-ll {ll0 - ll_te:.4f}')

        return trace, get_params(opt_state), log_freq


    def _get_arch(self, nlat):
        """Get TrainedArch by nlat value."""
        for a in self.archs:
            if a.nlat == nlat:
                return a
        # Fallback: reconstruct from old wbs
        idx = self.arch.index(nlat)
        return TrainedArch(nlat=nlat, wbs=self.wbs[idx],
                           history=self.history[idx] if idx < len(self.history) else {},
                           variational=self.variational)

    @property
    def arch(self):
        "Latent sizes for each trained architecture."
        if self.archs:
            return [a.nlat for a in self.archs]
        return [wb[0][0][0][1].size if self.variational else wb[0][0][1].size
                for wb in self.wbs]

    def _conf_rates_det(self, wbs, conns):
        @jax.jit
        def dist(a, b):
            return np.sum((a[:, None] - b) ** 2, axis=-1)
        crs = numpy.zeros((len(conns),) * 2)
        for i, ((ew, eb), _) in enumerate(wbs):
            u = conns[i] @ ew + eb
            for j, (_, (dw, db)) in enumerate(wbs):
                rec = u @ dw + db
                ok = dist(conns[j], rec).argmin(axis=1) == np.arange(conns[j].shape[0])
                crs[i, j] = 1 - ok.mean()
        return crs

    def calc_confusion_rate(self, arch, tts=None, self_recon_only=True):
        "Fraction of test subjects not fingerprinted correctly."
        tts = tts or self.tts
        wbs = self._get_arch(arch).wbs
        test_c = [c[tts:] for c in self.conns]
        if not self.variational:
            cr = self._conf_rates_det(wbs, test_c)
            return float(numpy.diag(cr).mean() if self_recon_only else cr.mean())

        @jax.jit
        def dist(a, b):
            return np.sum((a[:, None] - b) ** 2, axis=-1)
        total, count = 0.0, 0
        for i, (((w_mu, b_mu), _), _) in enumerate(wbs):
            mu = test_c[i] @ w_mu + b_mu
            for j, (_, (w_dec, b_dec)) in enumerate(wbs):
                if self_recon_only and i != j:
                    continue
                rec = mu @ w_dec + b_dec
                ok = np.argmin(dist(test_c[j], rec), axis=1) == np.arange(test_c[j].shape[0])
                total += 1.0 - np.mean(ok)
                count += 1
        return float(total / max(count, 1))


    def encode(self, arch, parc, tts=None, sample=False, key=None):
        "Encode the normalized connectomes of a view into latent space."
        ta = self._get_arch(arch)
        iparc = self.parcs.index(parc)
        c = self.conns[iparc] if tts is None else self.conns[iparc][tts:]
        if self.variational:
            ((w_mu, b_mu), (w_lv, b_lv)), _ = ta.wbs[iparc]
            mu = c @ w_mu + b_mu
            if not sample:
                return mu
            logvar = np.clip(c @ w_lv + b_lv, *_LOGVAR_CLIP)
            key = key if key is not None else jax.random.PRNGKey(SEED)
            return mu + np.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        (ew, eb), _ = ta.wbs[iparc]
        return c @ ew + eb

    def decode(self, arch, parc, z, raw=False):
        "Decode latent vectors into flat upper-tri connectomes."
        iparc = self.parcs.index(parc)
        _, (w_dec, b_dec) = self._get_arch(arch).wbs[iparc]
        rec = z @ w_dec + b_dec
        if raw:
            return rec
        return _denorm(rec, self.norm_types[iparc], self.means[iparc],
                       self.stds[iparc], self.scales[iparc], self.nonneg[iparc])

    def decode_conn(self, arch, parc, z, clip_positive=None):
        "Decode latents into full symmetric connectomes ``(ns, nn, nn)``."
        flat = self.decode(arch, parc, z, raw=False)
        if clip_positive is True:
            flat = np.maximum(flat, 0.0)
        return triu_to_mat(flat)

    def get_triu(self, parc, tts=None):
        "Return normalized flat upper-tri connectomes for a view."
        return self.conns[self.parcs.index(parc)][
            self.tts if tts is None else tts:]

    def get_conn(self, parc, tts=None):
        "Return empirical connectomes ``(ns, nn, nn)`` for a view."
        iparc = self.parcs.index(parc)
        flat = _denorm(self.get_triu(parc, tts), self.norm_types[iparc],
                       self.means[iparc], self.stds[iparc],
                       self.scales[iparc], self.nonneg[iparc])
        return triu_to_mat(flat)


    def calc_mvn(self, arch, tts=None):
        "Total-variance multivariate normal over the cohort latents."
        ta = self._get_arch(arch)
        tts = tts or self.tts
        us, var = [], []
        for i, wb in enumerate(ta.wbs):
            c = self.conns[i][tts:]
            if self.variational:
                ((w_mu, b_mu), (w_lv, b_lv)), _ = wb
                us.append(c @ w_mu + b_mu)
                var.append(np.exp(np.clip(c @ w_lv + b_lv, *_LOGVAR_CLIP)))
            else:
                (w_mu, b_mu), _ = wb
                us.append(c @ w_mu + b_mu)
        us = np.concatenate(us, axis=0)
        mean = np.mean(us, axis=0)
        cov = np.cov(us.T)
        if self.variational and var:
            cov = cov + np.diag(np.mean(np.concatenate(var, axis=0), axis=0))
        return MvNorm(us, mean, cov)

    def decompose_latent(self, arch, tts=None):
        "SVD of the centered cohort latents for a given architecture."
        mvn = self.calc_mvn(arch, tts)
        X = mvn.us - mvn.mean
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        return {'components': Vh,
                'explained_variance': (S ** 2) / (X.shape[0] - 1),
                'projected': X @ Vh.T,
                'mean': mvn.mean}


    @classmethod
    def combine(cls, cc1, cc2, shuffle=True):
        "Concatenate two CrossCoders with identical views and normalizations."
        assert cc1.variational == cc2.variational
        assert cc1.parcs == cc2.parcs
        cc = cls(variational=cc1.variational, chunked_training=cc1.chunked_training)
        cc.conns = [np.concatenate([a, b]) for a, b in zip(cc1.conns, cc2.conns)]
        if shuffle:
            n = cc.conns[0].shape[0]
            perm = jax.random.permutation(jax.random.PRNGKey(SEED), np.arange(n))
            cc.conns = [c[perm] for c in cc.conns]
        cc.means, cc.stds, cc.scales = cc1.means, cc1.stds, cc1.scales
        cc.norm_types, cc.nonneg = cc1.norm_types, cc1.nonneg
        cc.parcs = cc1.parcs
        if cc1.tts is not None and cc2.tts is not None:
            cc.tts = cc1.tts + cc2.tts
        else:
            cc.tts = cc.conns[0].shape[0] // 2
        return cc


def sweep_crosscoder(model, dims, n_trials=20, seed=42,
                     lr_range=(1e-5, 1e-2), niter_range=(500, 5000),
                     mb_choices=(32, 64, 128),
                     beta_end_range=(1e-6, 1e-3),
                     anneal_range=(500, 3000),
                     score_fn=None):
    """
    Random hyperparameter sweep over ``CrossCoder.train``.

    Trained weights are discarded after each trial; only summary statistics
    are kept.  Returns ``(results_sorted_by_score, best)``.
    """
    rng = numpy.random.default_rng(seed)
    score_fn = score_fn or (lambda cr, mse, std_min: cr)
    results = []
    for dim in dims:
        for _ in range(n_trials):
            lr = float(numpy.exp(rng.uniform(numpy.log(lr_range[0]), numpy.log(lr_range[1]))))
            niter = int(rng.integers(*niter_range))
            mb = int(rng.choice(mb_choices))
            if model.variational:
                beta_end = float(numpy.exp(
                    rng.uniform(numpy.log(beta_end_range[0]), numpy.log(beta_end_range[1]))))
                anneal = int(rng.integers(*anneal_range))
            else:
                beta_end, anneal = 0.0, 0

            n_wbs_before = len(model.wbs)
            try:
                trace, wbs, cr = model.train(
                    nlat=dim, lr=lr, niter=niter, mb=mb,
                    beta_start=0.0, beta_end=beta_end, anneal_steps=anneal)
                mu = numpy.asarray(model.encode(dim, model.parcs[0], sample=False))
                std_z = mu.std(axis=0)
                if model.variational:
                    final_mse = float(numpy.asarray(trace)[-1, 2])
                else:
                    final_mse = float(numpy.exp(trace[-1][1]))
                results.append({
                    'dim': dim, 'lr': lr, 'niter': niter, 'mb': mb,
                    'beta_end': beta_end, 'anneal_steps': anneal,
                    'cr': cr, 'mse': final_mse,
                    'latent_std_min': float(std_z.min()),
                    'latent_std_max': float(std_z.max()),
                    'score': score_fn(cr, final_mse, float(std_z.min())),
                })
            except Exception:
                while len(model.wbs) > n_wbs_before:
                    model.wbs.pop()
                    model.history.pop()
                    if model.archs:
                        model.archs.pop()
                continue
            model.wbs.pop()
            model.history.pop()
            if model.archs:
                model.archs.pop()

    results.sort(key=lambda r: r['score'])
    return results, (results[0] if results else None)
