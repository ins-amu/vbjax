"""
Unified Cross-Coder for amortized inference of whole-brain connectomes.

Supports both deterministic (AE) and variational (VAE) modes via a single
`CrossCoder` class. The variational mode is recommended when training data
originates from heterogeneous acquisition pipelines / cohorts, as the
stochastic latent space regularises against cohort-specific artefacts.
"""

import numpy as np
import pickle
import tqdm
import functools

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

SEED = 42

def _small(*sh):
    return jax.random.normal(jax.random.PRNGKey(SEED), sh) * 1e-3

def _init_weights(key, shape, scale=1.0):
    fan_in = shape[0]
    std = jnp.sqrt(scale / fan_in)
    return jax.random.normal(key, shape) * std

def triu_to_mat(triu):
    n = triu.shape[1]
    nn = int(np.ceil((1 + np.sqrt(1 + 8 * n)) / 2))
    i, j = jnp.triu_indices(nn, k=1)
    mat = jnp.zeros((triu.shape[0], nn, nn), 'f')
    return mat.at[:, i, j].set(triu).at[:, j, i].set(triu)

def triu_to_mat_np(triu):
    if triu.ndim == 1:
        n = triu.shape[0]
        nn = int(np.ceil((1 + np.sqrt(1 + 8 * n)) / 2))
        mat = np.zeros((nn, nn))
        i, j = np.triu_indices(nn, k=1)
        mat[i, j] = triu; mat[j, i] = triu
        return mat
    n_samples, n_feat = triu.shape
    nn = int(np.ceil((1 + np.sqrt(1 + 8 * n_feat)) / 2))
    mat = np.zeros((n_samples, nn, nn))
    i, j = np.triu_indices(nn, k=1)
    mat[:, i, j] = triu; mat[:, j, i] = triu
    return mat

class MvNorm:
    def __init__(self, us, u_mean, u_cov, key=None):
        self.us = us
        self.u_mean = u_mean
        self.u_cov = u_cov
        self.key = key or jax.random.PRNGKey(SEED)

    def sample(self, n):
        self.key, key = jax.random.split(self.key)
        return jax.random.multivariate_normal(key, self.u_mean, self.u_cov, shape=(n,))


class CrossCoder:

    def __init__(self, variational: bool = True, chunked_training: bool = True):
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

    def to_pkl(self, fname):
        stuff = {k: getattr(self, k) for k in
                 ['variational', 'chunked_training', 'conns', 'means', 'stds', 'scales',
                  'parcs', 'tts', 'wbs', 'history', 'norm_types', 'nonneg']}
        with open(fname, 'wb') as fd:
            pickle.dump(stuff, fd)

    @classmethod
    def from_pkl(cls, fname):
        with open(fname, 'rb') as fd:
            stuff = pickle.load(fd)
        self = cls(
            variational=stuff.get('variational', False),
            chunked_training=stuff.get('chunked_training', True)
        )
        for key in ['conns', 'means', 'stds', 'scales', 'parcs',
                    'tts', 'wbs', 'history', 'norm_types', 'nonneg']:
            if key in stuff:
                setattr(self, key, stuff[key])
        self.conns = [jnp.array(c) for c in self.conns]
        self.means = [jnp.array(m) for m in self.means]
        self.stds = [float(s) for s in self.stds]
        if not self.norm_types:
            self.norm_types = ['center' if s == 1.0 else 'zscore' for s in self.stds]
        if not self.scales:
            self.scales = [1.0] * len(self.conns)
        if not self.nonneg:
            self.nonneg = [False] * len(self.conns)
        return self

    @classmethod
    def from_numpy_array(cls, weights, tts=None, parc='Schaefer-17Networks',
                         variational=False, chunked_training=True, normalize='center'):
        weights = np.maximum(weights, 0.0)
        ns, nn, _ = weights.shape
        i, j = np.triu_indices(nn, k=1)
        triu = weights[:, i, j]
        self = cls(variational=variational, chunked_training=chunked_training)
        self.tts = tts if tts is not None else ns // 2
        self.add_view(triu, f'{nn}-{parc}', normalize=normalize, nonneg=True)
        return self

    def add_view(self, data, parc_name: str, normalize: str = 'zscore', nonneg: bool = False):
        data = jnp.array(data, dtype=jnp.float32)
        scale_val = 1.0
        std_val = 1.0

        if normalize == 'zscore':
            mu = jnp.mean(data, axis=0)
            centered = data - mu
            std_val = float(jnp.std(centered) + 1e-9)
            norm_data = centered / std_val
        elif normalize == 'center':
            mu = jnp.mean(data, axis=0)
            norm_data = data - mu
        elif normalize == 'logit':
            scale_val = float(jnp.max(data))
            eps = 1e-6
            x_scaled = (data / (scale_val + 1e-9)) * (1 - 2 * eps) + eps
            logits = jnp.log(x_scaled / (1 - x_scaled))
            mu = jnp.mean(logits, axis=0)
            std_val = float(jnp.std(logits) + 1e-9)
            norm_data = (logits - mu) / std_val
        else:
            mu = jnp.zeros(data.shape[1])
            norm_data = data

        self.conns.append(norm_data)
        self.means.append(mu)
        self.stds.append(std_val)
        self.scales.append(scale_val)
        self.parcs.append(parc_name)
        self.norm_types.append(normalize)
        self.nonneg.append(nonneg)

    def shuffle(self, seed=None):
        n = self.conns[0].shape[0]
        key = jax.random.PRNGKey(seed if seed is not None else SEED)
        perm = jax.random.permutation(key, jnp.arange(n))
        self.conns = [c[perm] for c in self.conns]
        return np.array(perm)

    def _make_wbs_det(self, nlat):
        wbs = []
        for c in self.conns:
            n = c.shape[1]
            w1, w2t = _small(2, n, nlat)
            b1 = _small(nlat)
            b2 = _small(n)
            wbs.append(((w1, b1), (w2t.T, b2)))
        return wbs

    def _make_wbs_var(self, nlat):
        wbs = []
        key = jax.random.PRNGKey(SEED)
        for c in self.conns:
            n_feat = c.shape[1]
            key, k1, k2, k3 = jax.random.split(key, 4)
            w_mu = _init_weights(k1, (n_feat, nlat))
            b_mu = jnp.zeros(nlat)
            w_lv = _init_weights(k2, (n_feat, nlat), scale=1e-4)
            b_lv = jnp.ones(nlat) * -10.0
            w_dec = _init_weights(k3, (nlat, n_feat))
            b_dec = jnp.zeros(n_feat)
            wbs.append((((w_mu, b_mu), (w_lv, b_lv)), (w_dec, b_dec)))
        return wbs

    def make_wbs(self, nlat):
        return self._make_wbs_var(nlat) if self.variational else self._make_wbs_det(nlat)

    def _make_loss_det(self):
        @jax.jit
        def loss_fn(wbs, conns):
            ll = 0.0
            for i, ((ew, eb), _) in enumerate(wbs):
                u = conns[i] @ ew + eb
                for j, (_, (dw, db)) in enumerate(wbs):
                    ll = ll + jnp.mean((u @ dw + db - conns[j]) ** 2)
            return ll
        grad_fn = jax.jit(jax.grad(loss_fn))
        return loss_fn, grad_fn

    def _make_loss_var(self):
        @jax.jit
        def loss_fn(wbs, conns, rng_key, beta):
            total_kl, total_recon = 0.0, 0.0
            recon_list = []
            for i, (((w_mu, b_mu), (w_lv, b_lv)), _) in enumerate(wbs):
                x_in = conns[i]
                mu = x_in @ w_mu + b_mu
                logvar = jnp.clip(x_in @ w_lv + b_lv, -15.0, 5.0)
                kl = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)
                total_kl += jnp.mean(kl)
                std = jnp.exp(0.5 * logvar)
                rng_key, subkey = jax.random.split(rng_key)
                z = mu + std * jax.random.normal(subkey, shape=mu.shape)
                for j, (_, (w_dec, b_dec)) in enumerate(wbs):
                    mse = jnp.mean(jnp.square(z @ w_dec + b_dec - conns[j]))
                    total_recon += mse
                    recon_list.append(mse)
            return total_recon + beta * total_kl, (total_recon, total_kl, recon_list)
        grad_fn = jax.jit(jax.grad(lambda w, c, k, b: loss_fn(w, c, k, b)[0]))
        return loss_fn, grad_fn

    def make_loss(self):
        return self._make_loss_var() if self.variational else self._make_loss_det()

    def train(self, nlat, lr=3e-4, niter=2000, tts=None, mb=64,
              beta_start=0.0, beta_end=0.001, anneal_steps=1500):
        tts = tts or self.tts
        if tts is None:
            raise ValueError("Set self.tts or pass tts= before training.")
        
        train_conns = [c[:tts] for c in self.conns]
        test_conns = [c[tts:] for c in self.conns]
        wbs = self.make_wbs(nlat)
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(wbs)
        loss_fn, grad_fn = self.make_loss()
        trace = []
        mbkey = jax.random.PRNGKey(mb)

        if self.variational:
            get_beta = lambda it: jnp.minimum(
                beta_start + (beta_end - beta_start) * (it / anneal_steps), beta_end)

            if self.chunked_training:
                log_freq = 50
                @functools.partial(jax.jit, static_argnums=(3,))
                def train_chunk(start_i, opt_state, mbkey, n_steps):
                    def body(carry, i):
                        opt_state, mbkey = carry
                        beta = get_beta(i)
                        mbkey, kb, kl = jax.random.split(mbkey, 3)
                        idx = jax.random.randint(kb, (mb,), 0, tts)
                        mb_c = [c[idx] for c in train_conns]
                        w = get_params(opt_state)
                        g = grad_fn(w, mb_c, kl, beta)
                        g = jax.tree.map(lambda x: jnp.clip(x, -5.0, 5.0), g)
                        opt_state = opt_update(i, g, opt_state)
                        return (opt_state, mbkey), None
                    (opt_state, mbkey), _ = jax.lax.scan(
                        body, (opt_state, mbkey), jnp.arange(n_steps) + start_i)
                    return opt_state, mbkey

                pbar = tqdm.tqdm(total=niter + 1)
                i = 0
                while i <= niter:
                    steps = min(log_freq, niter + 1 - i)
                    if steps == 0: break
                    opt_state, mbkey = train_chunk(i, opt_state, mbkey, steps)
                    beta = get_beta(i + steps - 1)
                    wbs = get_params(opt_state)
                    mbkey, _k = jax.random.split(mbkey)
                    log_idx = jax.random.randint(_k, (mb,), 0, tts)
                    log_mb = [c[log_idx] for c in train_conns]
                    l_tr, (r_tr, kl_tr, r_det) = loss_fn(wbs, log_mb, mbkey, beta)
                    l_te, (r_te, kl_te, _) = loss_fn(wbs, test_conns, mbkey, beta)
                    trace.append([float(x) for x in [l_tr, l_te, r_tr, kl_tr] + r_det])
                    pbar.set_description(f'β:{beta:.1e} R:{r_tr:.4f} KL:{kl_tr:.1f}')
                    i += steps; pbar.update(steps)
                pbar.close()
            else:
                log_freq = 1
                for i in (pbar := tqdm.trange(niter + 1)):
                    beta = get_beta(i)
                    mbkey, kb, kl = jax.random.split(mbkey, 3)
                    idx = jax.random.randint(kb, (mb,), 0, tts)
                    mb_c = [c[idx] for c in train_conns]
                    wbs = get_params(opt_state)
                    g = grad_fn(wbs, mb_c, kl, beta)
                    g = jax.tree.map(lambda x: jnp.clip(x, -5.0, 5.0), g)
                    opt_state = opt_update(i, g, opt_state)
                    
                    l_tr, (r_tr, kl_tr, r_det) = loss_fn(wbs, mb_c, mbkey, beta)
                    l_te, (r_te, kl_te, _) = loss_fn(wbs, test_conns, mbkey, beta)
                    trace.append([float(x) for x in [l_tr, l_te, r_tr, kl_tr] + r_det])
                    pbar.set_description(f'β:{beta:.1e} R:{r_tr:.4f} KL:{kl_tr:.1f}')
                
            wbs = get_params(opt_state)
            cr = self.calc_confusion_rate(nlat, tts=tts)
            self.wbs.append(wbs)
            self.history.append({'nlat': nlat, 'trace': trace, 'log_freq': log_freq, 'variational': True})
            return trace, wbs, cr

        else:
            if self.chunked_training:
                log_freq = 50
                @functools.partial(jax.jit, static_argnums=(3,))
                def train_chunk(start_i, opt_state, mbkey, n_steps):
                    def body(carry, i):
                        opt_state, mbkey = carry
                        mbkey, kb = jax.random.split(mbkey)
                        idx = jax.random.randint(kb, (mb,), 0, tts)
                        mb_c = [c[idx] for c in train_conns]
                        w = get_params(opt_state)
                        g = grad_fn(w, mb_c)
                        opt_state = opt_update(i, g, opt_state)
                        return (opt_state, mbkey), None
                    (opt_state, mbkey), _ = jax.lax.scan(
                        body, (opt_state, mbkey), jnp.arange(n_steps) + start_i)
                    return opt_state, mbkey

                pbar = tqdm.tqdm(total=niter + 1)
                i = 0
                while i <= niter:
                    steps = min(log_freq, niter + 1 - i)
                    if steps == 0: break
                    opt_state, mbkey = train_chunk(i, opt_state, mbkey, steps)
                    wbs = get_params(opt_state)
                    mbkey, _k = jax.random.split(mbkey)
                    log_idx = jax.random.randint(_k, (mb,), 0, tts)
                    log_mb = [c[log_idx] for c in train_conns]
                    
                    ll_tr = float(jnp.log(loss_fn(wbs, log_mb)))
                    ll_te = float(jnp.log(loss_fn(wbs, test_conns)))
                    trace.append((ll_tr, ll_te))
                    
                    if len(trace) == 1:
                        pbar.set_description('-ll 0.000')
                    else:
                        pbar.set_description(f'-ll {trace[0][1] - ll_te:0.3f}')
                        
                    i += steps; pbar.update(steps)
                pbar.close()
            else:
                log_freq = 1
                for i in (pbar := tqdm.trange(niter + 1)):
                    mbkey, _key = jax.random.split(mbkey)
                    idx = jax.random.randint(_key, (mb,), 0, tts)
                    mb_c = [c[idx] for c in train_conns]
                    wbs = get_params(opt_state)
                    ll_tr = float(jnp.log(loss_fn(wbs, mb_c)))
                    ll_te = float(jnp.log(loss_fn(wbs, test_conns)))
                    trace.append((ll_tr, ll_te))
                    pbar.set_description(f'-ll {trace[0][1] - ll_te:0.3f}')
                    opt_state = opt_update(i, grad_fn(wbs, mb_c), opt_state)

            wbs = get_params(opt_state)
            cr = self._all_conf_rates_det(wbs, test_conns).mean()
            self.wbs.append(wbs)
            self.history.append({'nlat': nlat, 'trace': trace, 'log_freq': log_freq, 'variational': False})
            return trace, wbs, cr

    def _all_conf_rates_det(self, wbs, conns):
        @jax.jit
        def dist(a, b):
            return jnp.sum((a[:, None] - b) ** 2, axis=-1)
        crs = np.zeros((len(conns),) * 2)
        for i, ((ew, eb), _) in enumerate(wbs):
            u = conns[i] @ ew + eb
            for j, (_, (dw, db)) in enumerate(wbs):
                rec = u @ dw + db
                ok = dist(conns[j], rec).argmin(axis=1) == jnp.r_[:conns[j].shape[0]]
                crs[i, j] = 1 - ok.mean()
        return crs

    def calc_confusion_rate(self, arch, tts=None, self_recon_only=True):
        tts = tts or self.tts
        iarch = self.arch.index(arch)
        wbs = self.wbs[iarch]
        @jax.jit
        def dist_mat(x, y):
            return jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        total_cr, count = 0.0, 0
        test_conns = [c[tts:] for c in self.conns]
        if self.variational:
            for i_enc, (((w_mu, b_mu), _), _) in enumerate(wbs):
                mu = test_conns[i_enc] @ w_mu + b_mu
                for i_dec, (_, (w_dec, b_dec)) in enumerate(wbs):
                    if self_recon_only and i_enc != i_dec: continue
                    rec = mu @ w_dec + b_dec
                    d = dist_mat(test_conns[i_dec], rec)
                    matches = jnp.argmin(d, axis=1) == jnp.arange(d.shape[0])
                    total_cr += (1.0 - jnp.mean(matches)); count += 1
        else:
            cr_mat = self._all_conf_rates_det(wbs, test_conns)
            if self_recon_only: return float(np.diag(cr_mat).mean())
            return float(cr_mat.mean())
        return float(total_cr / max(count, 1))

    @property
    def arch(self):
        arches = []
        for wb in self.wbs:
            if self.variational: arches.append(wb[0][0][0][1].size)
            else: arches.append(wb[0][0][1].size)
        return arches

    def encode(self, arch, parc, tts=None, sample=False):
        iarch, iparc = self.arch.index(arch), self.parcs.index(parc)
        c = self.conns[iparc][tts:] if tts is not None else self.conns[iparc]
        if self.variational:
            ((w_mu, b_mu), (w_lv, b_lv)), _ = self.wbs[iarch][iparc]
            mu = c @ w_mu + b_mu
            if not sample: return mu
            logvar = c @ w_lv + b_lv
            eps = jax.random.normal(jax.random.PRNGKey(SEED), shape=mu.shape)
            return mu + jnp.exp(0.5 * logvar) * eps
        else:
            (ew, eb), _ = self.wbs[iarch][iparc]
            return c @ ew + eb

    encode_conn = encode

    def decode(self, arch, parc, z, raw=False):
        iarch = self.arch.index(arch)
        iparc = self.parcs.index(parc)
        _, (w_dec, b_dec) = self.wbs[iarch][iparc]
        rec = z @ w_dec + b_dec
        if raw: return rec
        ntype = self.norm_types[iparc]
        mu = self.means[iparc]
        std_val = self.stds[iparc]
        if ntype == 'logit':
            logits = rec * std_val + mu
            probs = jax.nn.sigmoid(logits)
            scale_val = self.scales[iparc]
            eps = 1e-6
            out = (probs - eps) / (1 - 2 * eps) * scale_val
        else:
            out = rec * std_val + mu
        if self.nonneg[iparc]:
            out = jnp.maximum(out, 0.0)
        return out

    def decode_conn(self, parc, us, clip_positive=None):
        arch = us.shape[1]
        flat = self.decode(arch, parc, us, raw=False)
        if clip_positive is True:
            flat = jnp.maximum(flat, 0.0)
        return triu_to_mat(flat)

    def get_triu(self, parc, tts=None):
        return self.conns[self.parcs.index(parc)][tts if tts is not None else self.tts:]

    def get_conn(self, parc, tts=None):
        iparc = self.parcs.index(parc)
        c = self.get_triu(parc, tts)
        ntype = self.norm_types[iparc]
        mu = self.means[iparc]
        std_val = self.stds[iparc]
        if ntype == 'logit':
            logits = c * std_val + mu
            probs = jax.nn.sigmoid(logits)
            scale_val = self.scales[iparc]
            eps = 1e-6
            flat = (probs - eps) / (1 - 2 * eps) * scale_val
        else:
            flat = c * std_val + mu
        if self.nonneg[iparc]:
            flat = jnp.maximum(flat, 0.0)
        return triu_to_mat(flat)

    def calc_mvn(self, arch, tts=None):
        iarch = self.arch.index(arch)
        tts = tts or self.tts
        us_list = []
        var_list = []

        for i, wb in enumerate(self.wbs[iarch]):
            c = self.conns[i][tts:]
            if self.variational:
                ((w_mu, b_mu), (w_lv, b_lv)), _ = wb
                mu = c @ w_mu + b_mu
                logvar = jnp.clip(c @ w_lv + b_lv, -15.0, 5.0)
                us_list.append(mu)
                var_list.append(jnp.exp(logvar))
            else:
                (w_mu, b_mu), _ = wb
                us_list.append(c @ w_mu + b_mu)

        us = jnp.concatenate(us_list, axis=0)
        u_mean = jnp.mean(us, axis=0)
        cov_mu = jnp.cov(us.T)

        if self.variational and var_list:
            all_vars = jnp.concatenate(var_list, axis=0)
            mean_var = jnp.mean(all_vars, axis=0)
            cov_total = cov_mu + jnp.diag(mean_var)
        else:
            cov_total = cov_mu

        return MvNorm(us, u_mean, cov_total)

    def decompose_latent(self, arch, tts=None):
        mvn = self.calc_mvn(arch, tts)
        X = mvn.us - mvn.u_mean
        U, S, Vh = jnp.linalg.svd(X, full_matrices=False)
        return {'components': Vh, 'explained_variance': (S ** 2) / (X.shape[0] - 1),
                'projected': X @ Vh.T, 'mean': mvn.u_mean}

    @classmethod
    def combine(cls, cc1, cc2, shuffle=True):
        assert cc1.variational == cc2.variational
        assert cc1.parcs == cc2.parcs
        cc = cls(variational=cc1.variational, chunked_training=cc1.chunked_training)
        n1, n2 = cc1.conns[0].shape[0], cc2.conns[0].shape[0]
        nh = n1 + n2
        cc.conns = [jnp.concatenate([a, b]) for a, b in zip(cc1.conns, cc2.conns)]
        if shuffle:
            perm = jax.random.permutation(jax.random.PRNGKey(SEED), jnp.arange(nh))
            cc.conns = [c[perm] for c in cc.conns]
        cc.means = cc1.means; cc.stds = cc1.stds; cc.scales = cc1.scales
        cc.norm_types = cc1.norm_types; cc.nonneg = cc1.nonneg
        cc.parcs = cc1.parcs; cc.tts = nh // 2
        return cc

def sweep(model, dims, n_trials=20, seed=42,
          lr_range=(1e-5, 1e-2), niter_range=(500, 5000),
          mb_choices=(32, 64, 128),
          beta_end_range=(1e-6, 1e-3), anneal_range=(500, 3000),
          score_fn=None):
    rng = np.random.default_rng(seed)

    if score_fn is None:
        score_fn = lambda cr, mse, std_min: cr

    results = []
    total = len(dims) * n_trials

    for dim in dims:
        for trial in range(n_trials):
            lr = float(np.exp(rng.uniform(np.log(lr_range[0]), np.log(lr_range[1]))))
            niter = int(rng.integers(niter_range[0], niter_range[1]))
            mb = int(rng.choice(mb_choices))

            if model.variational:
                beta_end = float(np.exp(rng.uniform(np.log(beta_end_range[0]), np.log(beta_end_range[1]))))
                anneal_steps = int(rng.integers(anneal_range[0], anneal_range[1]))
            else:
                beta_end = 0.0
                anneal_steps = 0

            n_wbs_before = len(model.wbs)

            try:
                trace, wbs, cr = model.train(
                    nlat=dim, lr=lr, niter=niter, mb=mb,
                    beta_start=0.0, beta_end=beta_end, anneal_steps=anneal_steps,
                )
            except Exception:
                while len(model.wbs) > n_wbs_before:
                    model.wbs.pop()
                    model.history.pop()
                continue

            mu = np.array(model.encode(dim, model.parcs[0], sample=False))
            std_z = mu.std(axis=0)

            if model.variational:
                final_mse = float(np.array(trace)[-1][2])
            else:
                final_mse = float(np.exp(trace[-1][1]))

            score = score_fn(cr, final_mse, float(std_z.min()))

            res = {
                'dim': dim, 'lr': lr, 'niter': niter, 'mb': mb,
                'beta_end': beta_end, 'anneal_steps': anneal_steps,
                'cr': cr, 'mse': final_mse,
                'latent_std_min': float(std_z.min()),
                'latent_std_max': float(std_z.max()),
                'score': score,
            }
            results.append(res)
            model.wbs.pop()
            model.history.pop()

    results.sort(key=lambda r: r['score'])
    return results, results[0] if results else None