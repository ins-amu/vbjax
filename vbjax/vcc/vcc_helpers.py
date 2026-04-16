"""
Unified visualization helpers for CrossCoder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as plticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from sklearn.decomposition import PCA

from vcc_modules import triu_to_mat_np
from visualization_helpers import modify_axis_spines


def _clean_cbar(cbar, vmin, vmax, label=None, decimals=2, frameon=False, tick_length=0):
    cbar.outline.set_visible(frameon)
    cbar.ax.tick_params(length=tick_length, which='major')
    cbar.ax.tick_params(length=0, which='minor')
    cbar.ax.minorticks_off()

    ticks = [vmin, vmax]
    
    # Format labels robustly
    def _fmt(v):
        if abs(v) < 1e-9: return '0'
        if abs(v) >= 100: return f'{v:.0f}'
        if abs(v) >= 1: return f'{v:.{decimals}f}'
        return f'{v:.{max(decimals, 2)}g}'
    
    labels = [_fmt(vmin), _fmt(vmax)]
    if labels[0] == labels[1] and vmin != vmax:
        labels = [f'{vmin:.4g}', f'{vmax:.4g}']

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    if label:
        cbar.set_label(label, rotation=90, labelpad=10)


# ============================================================================
# Internal data accessors
# ============================================================================

def _get_latents(model, arch, parc, tts=None, sample=False):
    return np.array(model.encode(arch, parc, tts=tts, sample=sample))


def _get_empirical_flat(model, parc, tts=None):
    """Original-space flat upper-tri."""
    iparc = model.parcs.index(parc)
    start = tts if tts is not None else 0
    c = model.conns[iparc][start:]
    ntype = model.norm_types[iparc]
    mu = model.means[iparc]
    std_val = model.stds[iparc]
    if ntype == 'logit':
        logits = np.array(c) * std_val + np.array(mu)
        from scipy.special import expit
        probs = expit(logits)
        scale_val = model.scales[iparc]
        eps = 1e-6
        out = (probs - eps) / (1 - 2 * eps) * scale_val
    else:
        out = np.array(c) * std_val + np.array(mu)
    if model.nonneg[iparc]:
        out = np.maximum(out, 0.0)
    return out


def _get_flat_recon(model, arch, parc, latents):
    """Decoded flat upper-tri in original space (nonneg applied inside model.decode)."""
    return np.array(model.decode(arch, parc, latents, raw=False))


# ============================================================================
# Model selection traces
# ============================================================================

def plot_model_selection(results, ncols=3, figsize=None, savefig=False, text_ypos=0.6, text_xpos=0.95,
                         out_name='training_trace.svg', fig_save_loc=''):
    n = len(results)
    if n == 0: return plt.figure()
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None: figsize = (3.5 * ncols, 2.5 * nrows)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.4)

    for idx, res in enumerate(results):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        trace = np.array(res['trace'])
        dim = res.get('dim', '?')
        lf = res.get('log_freq', 1)
        iters = np.arange(len(trace)) * lf
        if trace.shape[1] == 2:
            # Deterministic: trace stores (log(MSE_train), log(MSE_test))
            # Plot directly — no exp+log round-trip
            ax.plot(iters, trace[:, 0], label='Train', lw=1.5, alpha=.9, c='slategray')
            ax.plot(iters, trace[:, 1], label='Test', lw=1.5, alpha=.9, c='salmon')
            ax.set_ylabel('log MSE')
        else:
            # Variational: trace stores raw losses
            ax.plot(iters, trace[:, 0], label='Train', lw=1, alpha=.8, c='slategray')
            ax.plot(iters, trace[:, 1], label='Test', lw=1.5, alpha=.8, c='salmon')
            ax.plot(iters, trace[:, 2], label='Recon', lw=1, alpha=.5, c='teal', ls='--')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
        ax.set_title(f'Dim {dim}'); ax.set_xlabel('Iteration')
        txt = f"Final: {trace[-1, 0]:.2e} / {trace[-1, 1]:.2e}"
        if 'cr' in res: txt += f"\nCR: {res['cr']:.4f}"
        ax.text(text_xpos, text_ypos, txt, transform=ax.transAxes, va='top', ha='right', fontsize=7)
        if idx == n - 1: ax.legend(frameon=False, fontsize=7)
        modify_axis_spines(ax, which=['x', 'y'], xticks_main=[0, iters[-1]])
    if savefig:
        fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
    return fig


# ============================================================================
# Identifiability
# ============================================================================

def plot_identifiability(model, arch, parc_name, n_subs=50, cmap='inferno',
                         savefig=False, out_name='identifiability.svg',
                         fig_save_loc='', suffix=''):
    if suffix: out_name = f'identifiability_{suffix}.svg'
    if isinstance(parc_name, list):
        nv = len(parc_name)
        fig, axes = plt.subplots(1, nv, figsize=(4 * nv, 4))
        if nv == 1: axes = [axes]
        for i, p in enumerate(parc_name):
            _plot_single_ident(model, arch, p, n_subs, cmap, axes[i])
        plt.tight_layout()
        if savefig: fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
        return fig
    fig, ax = plt.subplots(figsize=(4, 4))
    _plot_single_ident(model, arch, parc_name, n_subs, cmap, ax)
    if savefig: fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
    return fig


def _plot_single_ident(model, arch, parc, n_subs, cmap, ax):
    tts = getattr(model, 'tts', 0)
    Z = _get_latents(model, arch, parc, tts=tts)
    emp = _get_empirical_flat(model, parc, tts=tts)
    rec = _get_flat_recon(model, arch, parc, Z)

    n_test = emp.shape[0]
    ns = min(n_subs, n_test)
    
    # Random subset (not first-N)
    rng = np.random.default_rng()
    sel = np.sort(rng.choice(n_test, ns, replace=False))
    emp, rec = emp[sel], rec[sel]

    dist = np.sqrt(np.sum((emp[:, None, :] - rec[None, :, :]) ** 2, axis=-1))
    acc = np.mean(np.argmin(dist, axis=1) == np.arange(ns))

    im = ax.imshow(dist, origin='upper', cmap=cmap)
    ax.set_title(f'{parc}  Acc: {acc * 100:.1f}%')
    ax.set_xlabel('Reconstructed ID'); ax.set_ylabel('Ground truth ID')
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([0, ns - 1]); ax.set_yticks([0, ns - 1])
    ax.tick_params(length=0)
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    _clean_cbar(cb, float(dist.min()), float(dist.max()), label='Eucl. Dist')


# ============================================================================
# Fidelity
# ============================================================================

def plot_reconstruction_fidelity(model, arch, parc_name, cmap='inferno',
                                 gridsize=50, savefig=False,
                                 out_name='fidelity.svg', fig_save_loc='',
                                 suffix='', mask_zeros=True, hide_masked=True):
    """
    Hexbin plot of empirical vs reconstructed edge weights.
    
    Args:
        mask_zeros: If True, empirical zero-valued edges are excluded from
            the correlation and regression computation. These edges carry no
            structural information (tractography threshold artefacts) and bias
            the regression intercept upward. The fraction masked is reported.
        hide_masked: If True (and mask_zeros=True), the masked edges are not
            shown in the hexbin at all. If False, they are shown in the hexbin
            but excluded from the regression/correlation stats.
    """
    if suffix: out_name = f'fidelity_{suffix}.svg'
    if isinstance(parc_name, list):
        nv = len(parc_name)
        fig, axes = plt.subplots(1, nv, figsize=(4 * nv, 4))
        if nv == 1: axes = [axes]
        for i, p in enumerate(parc_name):
            _plot_single_fidelity(model, arch, p, cmap, gridsize, axes[i], mask_zeros, hide_masked)
        plt.tight_layout()
        if savefig: fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
        return fig
    fig, ax = plt.subplots(figsize=(4, 4))
    _plot_single_fidelity(model, arch, parc_name, cmap, gridsize, ax, mask_zeros, hide_masked)
    if savefig: fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
    return fig


def _plot_single_fidelity(model, arch, parc, cmap, gridsize, ax, mask_zeros=True, hide_masked=True):
    tts = getattr(model, 'tts', 0)
    Z = _get_latents(model, arch, parc, tts=tts)
    emp = _get_empirical_flat(model, parc, tts=tts)
    rec = _get_flat_recon(model, arch, parc, Z)

    # Report clipping stats (if nonneg was used)
    iparc = model.parcs.index(parc)
    if model.nonneg[iparc]:
        rec_norm = np.array(model.decode(arch, parc, Z, raw=True))
        ntype = model.norm_types[iparc]
        mu_np = np.array(model.means[iparc])
        std_val = model.stds[iparc]
        if ntype != 'logit':
            rec_unclipped = rec_norm * std_val + mu_np
            n_neg = np.sum(rec_unclipped < 0)
            pct_neg = 100 * n_neg / rec_unclipped.size
            if pct_neg > 0:
                print(f"[Fidelity {parc}] {pct_neg:.2f}% of decoded values were < 0 (clipped by nonneg)")

    x_all, y_all = emp.ravel(), rec.ravel()

    # Mask empirical zeros (tractography threshold artefacts)
    nonzero_mask = x_all > 0
    n_masked = np.sum(~nonzero_mask)
    pct_masked = 100 * n_masked / len(x_all)

    if mask_zeros and 'SC' in parc:
        x_fit, y_fit = x_all[nonzero_mask], y_all[nonzero_mask]
        print(f"[Fidelity {parc}] Masked {pct_masked:.1f}% zero-valued empirical edges from stats")
    else:
        x_fit, y_fit = x_all, y_all

    # Stats on non-zero edges only
    r_val = stats.spearmanr(x_fit, y_fit)[0]
    import warnings
    from numpy.polynomial.polyutils import RankWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RankWarning)
        m, b = np.polyfit(x_fit, y_fit, 1)

    # Hexbin: either all data or non-zero only
    if mask_zeros and hide_masked and 'SC' in parc:
        x_plot, y_plot = x_fit, y_fit
    else:
        x_plot, y_plot = x_all, y_all

    hb = ax.hexbin(x_plot, y_plot, gridsize=gridsize, cmap=cmap, mincnt=1, bins='log')

    # Data-dependent axis limits
    data_max = max(x_plot.max(), y_plot.max())
    if 'SC' in parc:
        ax_min, ax_max = 0, data_max * 1.05
    else:
        ax_min = min(x_plot.min(), y_plot.min()) * 1.05
        ax_max = data_max * 1.05
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)

    # Identity line (y = x)
    # ax.plot([ax_min, ax_max], [ax_min, ax_max], 'k--', lw=1, alpha=0.5)

    # Regression line
    xr = np.linspace(x_fit.min(), x_fit.max(), 100)
    ax.plot(xr, m * xr + b, 'k-', lw=2)

    # Annotation
    txt = f'$r={r_val:.2f}$\n$y={m:.2f}x{b:+.2f}$'

    # if mask_zeros and 'SC' in parc:
    #     txt += f'\n({pct_masked:.0f}% zeros masked)'

    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', fontweight='bold', fontsize=8)

    ax.set_xlabel('Empirical'); ax.set_ylabel('Reconstructed')
    ax.set_title(f'Fidelity ({parc})')

    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
    cb = plt.colorbar(hb, cax=cax)
    _clean_cbar(cb, *hb.get_clim(), label='Count (log)')
    modify_axis_spines(ax, which=['x', 'y'])


# ============================================================================
# Obs vs Pred
# ============================================================================

def plot_obs_vs_pred(model, arch, parc_name=None, n_subs=3, cmap='inferno', cmap_res='PuOr',
                     savefig=False, out_name='obs_vs_pred.svg',
                     fig_save_loc='', suffix='', subject_ids=None, start_idx=None):
    """
    Visual comparison of observed vs predicted connectivity matrices.

    Args:
        subject_ids: Array of subject ID strings (full dataset length).
        start_idx: Index to start drawing subjects from.
            - None: defaults to model.tts (test set only).
            - 0: draw from the entire dataset.
    """
    if suffix: out_name = f'obs_vs_pred_{suffix}.svg'
    if parc_name is None: parc_name = model.parcs[0]
    if isinstance(parc_name, list): parc_name = parc_name[0]

    # Determine slice
    if start_idx is not None:
        enc_start = start_idx
    else:
        enc_start = getattr(model, 'tts', 0)

    Z = _get_latents(model, arch, parc_name, tts=enc_start)
    n_avail = Z.shape[0]
    n_draw = min(n_subs, n_avail)
    idx = np.random.choice(n_avail, n_draw, replace=False)

    # Resolve subject IDs
    if subject_ids is not None:
        slice_ids = subject_ids[enc_start:]
    else:
        slice_ids = None

    n_rows = len(idx)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 3 * n_rows))
    if n_rows == 1: axes = axes[None, :]

    col_headers = ['Observation', 'Prediction', 'Residual']

    for row, i in enumerate(idx):
        emp = _get_empirical_flat(model, parc_name, tts=enc_start)[i]
        emp_mat = triu_to_mat_np(emp)
        rec = _get_flat_recon(model, arch, parc_name, Z[i:i + 1])[0]
        rec_mat = triu_to_mat_np(rec)
        diff = emp_mat - rec_mat

        r_val = np.corrcoef(emp.ravel(), rec.ravel())[0, 1]
        rmse = np.sqrt(np.mean((emp - rec) ** 2))
        mae = np.mean(np.abs(emp - rec))
        vmax = np.percentile(emp_mat, 99)
        vmin = 0 if 'SC' in parc_name else -vmax
        dmax = max(np.percentile(np.abs(diff), 99), 1e-9)

        plot_specs = [
            (emp_mat, vmin, vmax, cmap),
            (rec_mat, vmin, vmax, cmap),
            (diff, -dmax, dmax, cmap_res),
        ]

        subj_label = str(slice_ids[i]) if slice_ids is not None else f'#{enc_start + i}'

        for col, (mat, vm, vx, cm) in enumerate(plot_specs):
            ax = axes[row, col]
            im = ax.imshow(mat, cmap=cm, vmin=vm, vmax=vx)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)

            # Column headers on top row only
            if row == 0:
                title = col_headers[col]
                if col == 1: title += f'  ($r={r_val:.2f}$)'
                if col == 2: title += f'  (RMSE={rmse:.3f})'
                ax.set_title(title)
            else:
                if col == 1:
                    ax.text(0.5, 1.02, f'$r={r_val:.2f}$', transform=ax.transAxes,
                            ha='center', va='bottom', fontsize=9)
                if col == 2:
                    ax.text(0.5, 1.02, f'RMSE={rmse:.3f}', transform=ax.transAxes,
                            ha='center', va='bottom', fontsize=9)

            if col == 0:
                ax.set_ylabel(subj_label, rotation=90, labelpad=10)

            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _clean_cbar(cb, vm, vx)

    plt.tight_layout()
    if savefig:
        fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
    return fig


# ============================================================================
# Latent structure
# ============================================================================

_COHORT_MARKERS = {
    'OKB': 'o', 'HCA': '^', 'NKI': 's', 'OAS': 'D', 'CMC': 'v',
    'HCP': 'P', 'UKB': 'X', 'ADNI': '*', 'PPMI': 'h',
}
_FALLBACK_MARKERS = ['o', '^', 's', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p']


def plot_latent_structure(model, arch, parc_name, color_vals=None,
                          color_label=None, cohorts=None,
                          method='pca', n_components=2, tts=None,
                          subset_idx=None, dims_to_plot=(0, 1),
                          pca_params=None, umap_params=None,
                          start_idx=None, per_cohort_color=True,
                          savefig=False, out_name='latent.svg',
                          fig_save_loc='', suffix='', **kwargs):
    """
    Visualize latent space with PCA or UMAP.

    Args:
        start_idx: 0 for all subjects, None/model.tts for test set only.
        per_cohort_color: If True and cohorts are provided, color values are
            normalized within each cohort independently. This prevents
            cohorts with different age ranges from washing each other out.
            The colorbar reflects the reference cohort (first in sort order).
    """
    if suffix: out_name = f'latent_{suffix}.svg'

    if isinstance(parc_name, list):
        nv = len(parc_name)
        fig, axes = plt.subplots(1, nv, figsize=(5 * nv, 5))
        if nv == 1: axes = [axes]
        sc = None
        for i, p in enumerate(parc_name):
            cv = color_vals[i] if isinstance(color_vals, list) and len(color_vals) == nv else color_vals
            cl = color_label[i] if isinstance(color_label, list) and len(color_label) == nv else color_label
            sc_ = _plot_single_latent(model, arch, p, cv, cl, cohorts, method,
                                      n_components, tts, subset_idx, dims_to_plot,
                                      pca_params, umap_params, start_idx, per_cohort_color, kwargs, axes[i])
            if sc_: sc = sc_
        if sc is not None and color_vals is not None:
            plt.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
            cb = plt.colorbar(sc, cax=cbar_ax)
            _clean_cbar(cb, 0, 1, label=color_label if isinstance(color_label, str) else '')
        if savefig: fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
        return fig

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = _plot_single_latent(model, arch, parc_name, color_vals, color_label,
                             cohorts, method, n_components, tts, subset_idx,
                             dims_to_plot, pca_params, umap_params, start_idx,
                             per_cohort_color, kwargs, ax)
    if sc is not None and color_vals is not None:
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
        cb = plt.colorbar(sc, cax=cax)
        _clean_cbar(cb, 0, 1, label=color_label or '')
    if savefig: fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
    return fig


def _plot_single_latent(model, arch, parc, color_vals, color_label, cohorts,
                        method, n_components, tts, subset_idx, dims_to_plot,
                        pca_params, umap_params, start_idx, per_cohort_color, extra, ax):

    # Determine encoding start
    if start_idx is not None: enc_start = start_idx
    elif tts is not None: enc_start = tts
    else: enc_start = getattr(model, 'tts', 0)

    Z = _get_latents(model, arch, parc, tts=enc_start)

    if color_vals is not None:
        color_vals = np.asarray(color_vals)
        if len(color_vals) > Z.shape[0]: color_vals = color_vals[enc_start:]
    if cohorts is not None:
        cohorts = np.asarray(cohorts)
        if len(cohorts) > Z.shape[0]: cohorts = cohorts[enc_start:]

    if subset_idx is not None:
        Z = Z[subset_idx]
        if color_vals is not None: color_vals = color_vals[subset_idx]
        if cohorts is not None: cohorts = cohorts[subset_idx]

    # Color normalization
    if color_vals is not None:
        cv = np.asarray(color_vals, dtype=float)
        if per_cohort_color and cohorts is not None:
            # Normalize within each cohort independently
            cv_norm = np.zeros_like(cv)
            for coh in np.unique(cohorts):
                m = cohorts == coh
                cmin, cmax = cv[m].min(), cv[m].max()
                if cmax > cmin:
                    cv_norm[m] = (cv[m] - cmin) / (cmax - cmin)
                else:
                    cv_norm[m] = 0.5
            color_vals = cv_norm
        else:
            vmin, vmax = cv.min(), cv.max()
            color_vals = (cv - vmin) / (vmax - vmin) if vmax > vmin else cv * 0 + 0.5

    x_idx, y_idx = dims_to_plot

    if method == 'pca':
        pca_kw = pca_params or {}
        reducer = PCA(n_components=n_components, **pca_kw)
        emb = reducer.fit_transform(Z)
        lx = f'PC{x_idx + 1} ({reducer.explained_variance_ratio_[x_idx]:.1%})'
        ly = f'PC{y_idx + 1} ({reducer.explained_variance_ratio_[y_idx]:.1%})'
    elif method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")
        umap_kw = umap_params or {}
        umap_kw.setdefault('n_components', max(n_components, max(x_idx, y_idx) + 1))
        reducer = umap.UMAP(**umap_kw)
        emb = reducer.fit_transform(Z)
        lx, ly = f'UMAP{x_idx + 1}', f'UMAP{y_idx + 1}'
    else:
        raise ValueError(f"Unknown method '{method}'")

    sc = None
    scatter_kw = {k: v for k, v in extra.items() if k in ('s', 'alpha')}
    scatter_kw.setdefault('s', 15)
    scatter_kw.setdefault('alpha', 0.8)

    if cohorts is not None:
        unique_cohorts = np.unique(cohorts)
        for ci, coh in enumerate(unique_cohorts):
            m = cohorts == coh
            mk = _COHORT_MARKERS.get(coh, _FALLBACK_MARKERS[ci % len(_FALLBACK_MARKERS)])
            c = color_vals[m] if color_vals is not None else None
            sc_ = ax.scatter(emb[m, x_idx], emb[m, y_idx], c=c, marker=mk,
                             label=coh, cmap='inferno', edgecolor='k', lw=0.3,
                             vmin=0, vmax=1, **scatter_kw)
            if c is not None: sc = sc_
        ax.legend(frameon=False, markerscale=1.5, fontsize=8)
    else:
        sc = ax.scatter(emb[:, x_idx], emb[:, y_idx], c=color_vals, cmap='viridis',
                        edgecolor='k', lw=0.3, vmin=0, vmax=1, **scatter_kw)

    ax.set_xlabel(lx); ax.set_ylabel(ly)
    ax.set_title(f'Latent ({parc})')
    modify_axis_spines(ax, which=['x', 'y'])
    return sc


# ============================================================================
# Generative validation & traversal
# ============================================================================

def validate_generative_stats(model, arch, parc_name, n_samples=500,
                              savefig=False, out_name='gen_stats.svg',
                              fig_save_loc='', alpha=0.85,
                              c_emp='slategray', c_gen='tan'):
    tts = getattr(model, 'tts', 0)
    emp = _get_empirical_flat(model, parc_name, tts=tts)
    mvn = model.calc_mvn(arch, tts=tts)
    z_gen = np.array(mvn.sample(n_samples))
    rec = _get_flat_recon(model, arch, parc_name, z_gen)
    mu_e, mu_g = np.mean(emp, 1), np.mean(rec, 1)
    mx_e, mx_g = np.max(emp, 1), np.max(rec, 1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].hist(mu_e, 30, alpha=alpha, density=True, label='Empirical', color=c_emp)
    axes[0].hist(mu_g, 30, alpha=alpha, density=True, label='Generated', color=c_gen)
    axes[0].set_title('Mean Edge Weight'); axes[0].legend(frameon=False)
    modify_axis_spines(axes[0])
    axes[1].hist(mx_e, 30, alpha=alpha, density=True, label='Empirical', color=c_emp)
    axes[1].hist(mx_g, 30, alpha=alpha, density=True, label='Generated', color=c_gen)
    axes[1].set_title('Max Edge Weight')
    modify_axis_spines(axes[1])
    ks = stats.ks_2samp(mu_e, mu_g)
    fig.suptitle(f'Generative Validation (KS p={ks.pvalue:.2e})')
    if savefig: fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
    return fig


def plot_latent_traversal(model, arch, parc_name, dim_idx=0, range_sd=3.0,
                          n_steps=7, savefig=False, out_name='traversal.svg',
                          fig_save_loc=''):
    vals = np.linspace(-range_sd, range_sd, n_steps)
    z = np.zeros((n_steps, arch))
    z[:, dim_idx] = vals
    rec = _get_flat_recon(model, arch, parc_name, z)
    mats = triu_to_mat_np(rec)
    vmax = np.percentile(mats, 99.9)
    vmin = np.percentile(mats, 0.1)
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2.5))
    im = None
    for i in range(n_steps):
        im = axes[i].imshow(mats[i], cmap='inferno', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'z[{dim_idx}]={vals[i]:.1f}'); axes[i].axis('off')
    cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
    cb = plt.colorbar(im, cax=cax)
    _clean_cbar(cb, vmin, vmax)
    if savefig: fig.savefig(fig_save_loc + out_name, transparent=True, bbox_inches='tight')
    return fig