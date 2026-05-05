"""
Visualisation helpers for :class:`~vbjax.crosscoder.CrossCoder` models.

Matplotlib is imported lazily so the core package has no hard dependency.
All plotting routines expect a trained ``CrossCoder`` with at least one
architecture in ``model.wbs``.
"""

import numpy
from scipy import stats

from .crosscoder import triu_to_mat_np


_COHORT_MARKERS = {'OKB': 'o', 'HCA': '^', 'NKI': 's', 'OAS': 'D', 'CMC': 'v',}
_FALLBACK_MARKERS = ['o', '^', 's', 'D', 'v']


def _trim_spines(ax, which=('x', 'y')):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if 'x' in which:
        ax.spines['bottom'].set_bounds(*ax.get_xlim())
    if 'y' in which:
        ax.spines['left'].set_bounds(*ax.get_ylim())


def _clean_cbar(cbar, vmin, vmax, label=None, decimals=2):
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, which='both')
    cbar.ax.minorticks_off()

    def fmt(v):
        if abs(v) < 1e-9: return '0'
        if abs(v) >= 100: return f'{v:.0f}'
        if abs(v) >= 1: return f'{v:.{decimals}f}'
        return f'{v:.{max(decimals, 2)}g}'

    labels = [fmt(vmin), fmt(vmax)]
    if labels[0] == labels[1] and vmin != vmax:
        labels = [f'{vmin:.4g}', f'{vmax:.4g}']
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(labels)
    if label:
        cbar.set_label(label, rotation=90, labelpad=10)


def _latents(model, arch, parc, tts=None, sample=False):
    return numpy.asarray(model.encode(arch, parc, tts=tts, sample=sample))


def _empirical_flat(model, parc, tts=None):
    from .crosscoder import _denorm
    iparc = model.parcs.index(parc)
    c = model.conns[iparc][(tts or 0):]
    return numpy.asarray(_denorm(
        c, model.norm_types[iparc], model.means[iparc],
        model.stds[iparc], model.scales[iparc], model.nonneg[iparc]))


def _flat_recon(model, arch, parc, latents):
    return numpy.asarray(model.decode(arch, parc, latents, raw=False))


def plot_training(results, ncols=3, figsize=None):
    "Plot train/test traces for a list of training results."
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(results)
    if n == 0:
        return plt.figure()
    ncols = min(ncols, n)
    nrows = int(numpy.ceil(n / ncols))
    fig = plt.figure(figsize=figsize or (3.5 * ncols, 2.5 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.4)

    for idx, res in enumerate(results):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        trace = numpy.asarray(res['trace'])
        iters = numpy.arange(len(trace)) * res.get('log_freq', 1)
        if trace.shape[1] == 2:
            ax.plot(iters, trace[:, 0], label='train', lw=1.5, c='slategray')
            ax.plot(iters, trace[:, 1], label='test', lw=1.5, c='salmon')
            ax.set_ylabel('log MSE')
        else:
            ax.plot(iters, trace[:, 0], label='train', lw=1.0, c='slategray')
            ax.plot(iters, trace[:, 1], label='test', lw=1.5, c='salmon')
            ax.plot(iters, trace[:, 2], label='recon', lw=1.0, c='teal', ls='--')
            ax.set_yscale('log')
            ax.set_ylabel('loss')
        ax.set_title(f"dim {res.get('dim', '?')}")
        ax.set_xlabel('iter')
        info = f"final: {trace[-1, 0]:.2e} / {trace[-1, 1]:.2e}"
        if 'cr' in res:
            info += f"\nCR: {res['cr']:.3f}"
        ax.text(0.95, 0.6, info, transform=ax.transAxes, ha='right', va='top', fontsize=7)
        if idx == n - 1:
            ax.legend(frameon=False, fontsize=7)
        _trim_spines(ax)
    return fig


def plot_identifiability(model, arch, parc, n_subs=50, cmap='inferno', ax=None):
    "Pairwise empirical/reconstructed distances on the test set."
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure
    tts = getattr(model, 'tts', 0)
    Z = _latents(model, arch, parc, tts=tts)
    emp = _empirical_flat(model, parc, tts=tts)
    rec = _flat_recon(model, arch, parc, Z)

    n_test = emp.shape[0]
    ns = min(n_subs, n_test)
    sel = numpy.sort(numpy.random.default_rng().choice(n_test, ns, replace=False))
    emp, rec = emp[sel], rec[sel]

    dist = numpy.sqrt(numpy.sum((emp[:, None] - rec[None]) ** 2, axis=-1))
    acc = numpy.mean(numpy.argmin(dist, axis=1) == numpy.arange(ns))

    im = ax.imshow(dist, cmap=cmap)
    ax.set_title(f'{parc}  acc: {acc * 100:.1f}%')
    ax.set_xlabel('reconstructed'); ax.set_ylabel('ground truth')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0, ns - 1]); ax.set_yticks([0, ns - 1])
    ax.tick_params(length=0)
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
    _clean_cbar(plt.colorbar(im, cax=cax), float(dist.min()), float(dist.max()),
                label='eucl. dist')
    return fig


def plot_fidelity(model, arch, parc, cmap='inferno', gridsize=50,
                  mask_zeros=True, hide_masked=True, ax=None):
    "Hexbin of empirical vs reconstructed edge weights with regression line."
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure
    tts = getattr(model, 'tts', 0)
    Z = _latents(model, arch, parc, tts=tts)
    emp = _empirical_flat(model, parc, tts=tts)
    rec = _flat_recon(model, arch, parc, Z)

    iparc = model.parcs.index(parc)
    x, y = emp.ravel(), rec.ravel()
    keep = (x > 0) if (mask_zeros and model.nonneg[iparc]) else numpy.ones_like(x, dtype=bool)
    xf, yf = x[keep], y[keep]
    r_val = stats.spearmanr(xf, yf)[0]
    m, b = numpy.polyfit(xf, yf, 1)

    xp, yp = (xf, yf) if (mask_zeros and hide_masked and model.nonneg[iparc]) else (x, y)
    hb = ax.hexbin(xp, yp, gridsize=gridsize, cmap=cmap, mincnt=1, bins='log')
    dmax = max(xp.max(), yp.max())
    dmin = 0 if model.nonneg[iparc] else min(xp.min(), yp.min()) * 1.05
    ax.set_xlim(dmin, dmax * 1.05); ax.set_ylim(dmin, dmax * 1.05)

    xr = numpy.linspace(xf.min(), xf.max(), 100)
    ax.plot(xr, m * xr + b, 'k-', lw=2)
    ax.text(0.05, 0.95, f'$r={r_val:.2f}$\n$y={m:.2f}x{b:+.2f}$',
            transform=ax.transAxes, va='top', fontweight='bold', fontsize=8)
    ax.set_xlabel('empirical'); ax.set_ylabel('reconstructed')
    ax.set_title(f'fidelity ({parc})')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
    _clean_cbar(plt.colorbar(hb, cax=cax), *hb.get_clim(), label='count (log)')
    _trim_spines(ax)
    return fig


def plot_obs_vs_pred(model, arch, parc=None, n_subs=3, cmap='inferno',
                     cmap_res='PuOr', subject_ids=None, start_idx=None):
    "Observed/predicted/residual panels for a few random test subjects."
    import matplotlib.pyplot as plt

    parc = parc or model.parcs[0]
    iparc = model.parcs.index(parc)
    enc_start = start_idx if start_idx is not None else getattr(model, 'tts', 0)
    Z = _latents(model, arch, parc, tts=enc_start)
    n_draw = min(n_subs, Z.shape[0])
    idx = numpy.random.choice(Z.shape[0], n_draw, replace=False)
    slice_ids = subject_ids[enc_start:] if subject_ids is not None else None

    fig, axes = plt.subplots(n_draw, 3, figsize=(10, 3 * n_draw), squeeze=False)
    headers = ['observation', 'prediction', 'residual']
    for row, i in enumerate(idx):
        emp = _empirical_flat(model, parc, tts=enc_start)[i]
        rec = _flat_recon(model, arch, parc, Z[i:i + 1])[0]
        emp_m, rec_m = triu_to_mat_np(emp), triu_to_mat_np(rec)
        diff = emp_m - rec_m
        r_val = numpy.corrcoef(emp, rec)[0, 1]
        rmse = numpy.sqrt(numpy.mean((emp - rec) ** 2))
        vmax = numpy.percentile(emp_m, 99)
        vmin = 0 if model.nonneg[iparc] else -vmax
        dmax = max(numpy.percentile(numpy.abs(diff), 99), 1e-9)
        panels = [(emp_m, vmin, vmax, cmap),
                  (rec_m, vmin, vmax, cmap),
                  (diff, -dmax, dmax, cmap_res)]
        label = str(slice_ids[i]) if slice_ids is not None else f'#{enc_start + i}'
        for col, (mat, vn, vx, cm) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(mat, cmap=cm, vmin=vn, vmax=vx)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(headers[col])
            if col == 1:
                ax.set_xlabel(f'$r={r_val:.2f}$', fontsize=8)
            if col == 2:
                ax.set_xlabel(f'RMSE={rmse:.3f}', fontsize=8)
            if col == 0:
                ax.set_ylabel(label, rotation=90, labelpad=10)
            _clean_cbar(plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04), vn, vx)
    fig.tight_layout()
    return fig


def plot_latent(model, arch, parc, color_vals=None, cohorts=None,
                method='pca', n_components=2, tts=None, dims=(0, 1),
                per_cohort_color=True, pca_params=None, umap_params=None,
                cohort_markers=None, fallback_markers=None,
                ax=None, **scatter_kw):
    "2-D latent scatter via PCA or UMAP, optionally colored by a covariate."
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.decomposition import PCA

    enc_start = tts if tts is not None else getattr(model, 'tts', 0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    Z = _latents(model, arch, parc, tts=enc_start)

    if color_vals is not None:
        color_vals = numpy.asarray(color_vals)
        if len(color_vals) > Z.shape[0]:
            color_vals = color_vals[enc_start:]
    if cohorts is not None:
        cohorts = numpy.asarray(cohorts)
        if len(cohorts) > Z.shape[0]:
            cohorts = cohorts[enc_start:]

    if color_vals is not None:
        cv = color_vals.astype(float)
        if per_cohort_color and cohorts is not None:
            norm = numpy.zeros_like(cv)
            for coh in numpy.unique(cohorts):
                m = cohorts == coh
                lo, hi = cv[m].min(), cv[m].max()
                norm[m] = (cv[m] - lo) / (hi - lo) if hi > lo else 0.5
            color_vals = norm
        else:
            lo, hi = cv.min(), cv.max()
            color_vals = (cv - lo) / (hi - lo) if hi > lo else cv * 0 + 0.5

    if method == 'pca':
        reducer = PCA(n_components=n_components, **(pca_params or {}))
        emb = reducer.fit_transform(Z)
        lx = f'PC{dims[0] + 1} ({reducer.explained_variance_ratio_[dims[0]]:.1%})'
        ly = f'PC{dims[1] + 1} ({reducer.explained_variance_ratio_[dims[1]]:.1%})'
    elif method == 'umap':
        import umap
        kw = dict(umap_params or {})
        kw.setdefault('n_components', max(n_components, max(dims) + 1))
        reducer = umap.UMAP(**kw)
        emb = reducer.fit_transform(Z)
        lx, ly = f'UMAP{dims[0] + 1}', f'UMAP{dims[1] + 1}'
    else:
        raise ValueError(f"unknown method '{method}'")

    scatter_kw.setdefault('s', 15)
    scatter_kw.setdefault('alpha', 0.8)
    sc = None
    if cohorts is not None:
        for ci, coh in enumerate(numpy.unique(cohorts)):
            m = cohorts == coh
            _markers = cohort_markers if cohort_markers is not None else _COHORT_MARKERS
            _fallback = fallback_markers if fallback_markers is not None else _FALLBACK_MARKERS
            mk = _markers.get(coh, _fallback[ci % len(_fallback)])
            c = color_vals[m] if color_vals is not None else None
            sc_ = ax.scatter(emb[m, dims[0]], emb[m, dims[1]], c=c, marker=mk,
                             label=coh, cmap='inferno', edgecolor='k', lw=0.3,
                             vmin=0, vmax=1, **scatter_kw)
            if c is not None:
                sc = sc_
        ax.legend(frameon=False, markerscale=1.5, fontsize=8)
    else:
        sc = ax.scatter(emb[:, dims[0]], emb[:, dims[1]], c=color_vals, cmap='viridis',
                        edgecolor='k', lw=0.3, vmin=0, vmax=1, **scatter_kw)

    ax.set_xlabel(lx); ax.set_ylabel(ly); ax.set_title(f'latent ({parc})')
    _trim_spines(ax)
    if sc is not None and color_vals is not None:
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
        _clean_cbar(plt.colorbar(sc, cax=cax), 0, 1)
    return fig


def plot_generative(model, arch, parc, n_samples=500, alpha=0.85,
                    c_emp='slategray', c_gen='tan'):
    "Compare empirical and model-sampled edge-weight summaries."
    import matplotlib.pyplot as plt

    tts = getattr(model, 'tts', 0)
    emp = _empirical_flat(model, parc, tts=tts)
    z_gen = numpy.asarray(model.calc_mvn(arch, tts=tts).sample(n_samples))
    rec = _flat_recon(model, arch, parc, z_gen)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, emp_stat, gen_stat, title in [
            (axes[0], emp.mean(1), rec.mean(1), 'mean edge weight'),
            (axes[1], emp.max(1), rec.max(1), 'max edge weight')]:
        ax.hist(emp_stat, 30, alpha=alpha, density=True, label='empirical', color=c_emp)
        ax.hist(gen_stat, 30, alpha=alpha, density=True, label='generated', color=c_gen)
        ax.set_title(title)
        _trim_spines(ax)
    axes[0].legend(frameon=False)
    ks = stats.ks_2samp(emp.mean(1), rec.mean(1))
    fig.suptitle(f'generative validation (KS p={ks.pvalue:.2e})')
    return fig


def plot_traversal(model, arch, parc, dim_idx=0, range_sd=3.0, n_steps=7):
    "Sweep a single latent dimension and visualise the decoded connectome."
    import matplotlib.pyplot as plt

    vals = numpy.linspace(-range_sd, range_sd, n_steps)
    z = numpy.zeros((n_steps, arch))
    z[:, dim_idx] = vals
    mats = triu_to_mat_np(_flat_recon(model, arch, parc, z))
    vmax = numpy.percentile(mats, 99.9)
    vmin = numpy.percentile(mats, 0.1)
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2.5))
    im = None
    for i in range(n_steps):
        im = axes[i].imshow(mats[i], cmap='inferno', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'z[{dim_idx}]={vals[i]:.1f}')
        axes[i].axis('off')
    cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
    _clean_cbar(plt.colorbar(im, cax=cax), vmin, vmax)
    return fig
