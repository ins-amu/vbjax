# %%
import os
import gc
import importlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import vcc_modules as vcc
import vcc_helpers as vch

reload_all = lambda: [importlib.reload(m) for m in [vcc, vch]]

plt.rcParams['svg.fonttype'] = 'none'
fontfamily = "sans-serif"
fontsize = 10
plt.rc('font', size=fontsize, family=fontfamily)
plt.rcParams.update({'axes.titlesize': plt.rcParams['font.size']})

# %%
proj_dir = '/home/ahesmaeili/git/gast_dopa/'
okb_dir = '/home/ahesmaeili/Documents/DATA/gast_aging/empirical/1000brains/'
hca_dir = '/home/ahesmaeili/Documents/DATA/gast_aging/empirical/hca/'
data_dir = '/home/ahesmaeili/Documents/DATA/gast_aging/'
proj_data_dir = proj_dir + 'data/'
fig_save_loc = proj_dir + 'figs/'

sc_array_dir = proj_data_dir + 'empirical/connectivity/'
fc_array_dir = proj_data_dir + 'empirical/functional/'

# %% [markdown]
# # Training XCODER

# %%
reload_all()
# del(xc); gc.collect()

# %%
norm_method = 'log'

sc_training_array = xr.load_dataarray(os.path.join(sc_array_dir, f'cohort_scs_normalized_{norm_method}.nc'))

# Shuffle so that train/test sets have subjects from all cohorts
shuffled_indices = np.random.permutation(len(sc_training_array.subject))
sc_training_array = sc_training_array.isel(subject=shuffled_indices)

sc_training_array = sc_training_array / sc_training_array.max()

n_nodes = sc_training_array.shape[1]
sc_triu_inds  = np.triu_indices(n_nodes, k=1)
sc_array_flat = np.array([sc_training_array[i].values[sc_triu_inds] for i in range(sc_training_array.shape[0])])

cohort_labels = sc_training_array.cohort.values
age_array = sc_training_array.age.values
subject_ids = sc_training_array.subject.values

# %%
vcc_vari = False
vxc = vcc.CrossCoder(variational=vcc_vari)

# normalize = 'logit'
normalize = 'zscore' if vcc_vari else 'center'
mode_tag = 'vxc' if vcc_vari else 'xc'
vxc_save_path = os.path.join(sc_array_dir, f'{mode_tag}_cohort_aging_{norm_method}.pkl')

# %%
vxc.add_view(sc_array_flat, 'SC', normalize=normalize, nonneg=True)

# perm = vxc.shuffle()
# cohort_labels = cohort_labels[perm]
# age_array = age_array[perm]
# subject_ids = subject_ids[perm]

vxc.tts = int(sc_training_array.shape[0] * 0.8)
print(f"Train/test split: {vxc.tts} \ {sc_array_flat.shape[0] - vxc.tts}")
print(f"  Train cohorts: {np.unique(cohort_labels[:vxc.tts], return_counts=True)}")
print(f"  Test  cohorts: {np.unique(cohort_labels[vxc.tts:], return_counts=True)}")

# %%
# reload_all()
# sweep_results, best = vcc.sweep(vxc, dims=[3, 4], n_trials=20, lr_range=(5e-4, 5e-3), beta_end_range=(1e-6, 1e-5), niter=25000)

# %%
if vcc_vari:
    dims   = [2, 3, 4]
    niter  = 25000
    lr     = 8e-4
    beta_end = 1e-6
    anneal   = 2200
else:
    dims   = [2, 3, 4]
    niter  = 30000
    lr     = 5e-5
    beta_end = 0.0
    anneal   = 0

results = []
for dim in dims:
    print(f"\n{'='*40}  Dim {dim}  {'='*40}")
    trace, wbs, cr = vxc.train(
        nlat=dim, niter=niter, lr=lr, mb=64,
        beta_start=0.0, beta_end=beta_end, anneal_steps=anneal,
    )

    mu = vxc.encode(dim, 'SC', sample=False)
    std_z = np.array(mu).std(axis=0)
    print(f"CR: {cr:.4f}  |  Latent std range: [{std_z.min():.3f}, {std_z.max():.3f}]")

    res = {
        'dim': dim, 'cr': cr,
        'mse': float(np.array(trace)[-1, 2]) if vcc_vari else np.exp(trace[-1][1]),
        'trace': np.array(trace),
        'log_freq': vxc.history[-1]['log_freq'],
    }
    results.append(res)

# vxc.to_pkl(vxc_save_path)

# %%
vxc = vcc.CrossCoder(variational=vcc_vari)
vxc = vxc.from_pkl(vxc_save_path)
savefig = False
best_dim = dims[-1]

# dweights_array = xr.load_dataarray(sc_array_dir + f'cohort_scs_log_decoded_{best_dim}dim.nc')
# cohort_labels = dweights_array.cohort.to_numpy()

# %%
best_dim = vxc.arch[-1]

fig_trace = vch.plot_model_selection(results, text_ypos=0.65, savefig=True, fig_save_loc=fig_save_loc)

# %%
fig_fid = vch.plot_reconstruction_fidelity(vxc, arch=best_dim, parc_name='SC', gridsize=30, mask_zeros=True, hide_masked=True,
                                           fig_save_loc=fig_save_loc, savefig=savefig, suffix=f'{mode_tag}_sc_{best_dim}dim',)
# fig_fid.savefig('/home/ahesmaeili/Desktop/test.jpg', dpi=400)

# %%
fig_ident = vch.plot_identifiability(vxc, arch=best_dim, parc_name='SC', n_subs=200, fig_save_loc=fig_save_loc, savefig=savefig,
                                     suffix=f'{mode_tag}_ident_sc_{best_dim}dim')

# %%
fig_obs = vch.plot_obs_vs_pred(vxc, arch=best_dim, parc_name='SC', n_subs=5, cmap_res='PuOr_r', subject_ids=subject_ids, start_idx=0,
                               fig_save_loc=fig_save_loc, savefig=False, suffix=f'{mode_tag}_sc_{best_dim}dim',)

# %%
_ = vch.plot_latent_structure(vxc, arch=best_dim, parc_name='SC',
                            cohorts=cohort_labels,
                            start_idx=0, per_cohort_color=True,
                            method='pca', n_components=best_dim,
                            dims_to_plot=(0, 1), s=15, fig_save_loc=fig_save_loc,
                            savefig=True, suffix=f'{mode_tag}_latent_age_{best_dim}dim')

# %%
_ = vch.plot_latent_structure(vxc, arch=best_dim, parc_name='SC',
                            cohorts=cohort_labels,
                            start_idx=0, per_cohort_color=True,
                            method='umap', n_components=best_dim,
                            umap_params={'n_neighbors': 50, 'min_dist': 0.7},
                            dims_to_plot=(0, 1), s=15,
                            savefig=False, suffix=f'{mode_tag}_latent_umap_{best_dim}dim')

# %%
# _ = vch.plot_latent_structure(vxc, arch=best_dim, parc_name='SC',
#                             cohorts=cohort_labels,
#                             start_idx=0, per_cohort_color=True,
#                             method='umap', n_components=best_dim,
#                             umap_params={'n_neighbors': 50, 'min_dist': 0.7},
#                             dims_to_plot=(0, 1), s=15,
#                             savefig=False, suffix=f'{mode_tag}_latent_umap_{best_dim}dim')

# %%
fig_gen = vch.validate_generative_stats(vxc, arch=best_dim, parc_name='SC',
                                        n_samples=500, savefig=False)

# %%
fig_trav = vch.plot_latent_traversal(vxc, arch=best_dim, parc_name='SC',
                                     dim_idx=0, savefig=False)

# %% [markdown]
# # Saving decoded weights

# %%
# del(vxc); gc.collect()

vxc = vcc.CrossCoder.from_pkl(sc_array_dir + f'xc_cohort_aging_log.pkl')
parc = vxc.parcs[0]
arch = 3

u = vxc.encode(arch, parc, tts=0)
weights_decoded = vxc.decode_conn(parc, u)

# %%
dweights_array = sc_training_array.copy()
dweights_array[...] = np.array(weights_decoded)
dweights_array.to_netcdf(sc_array_dir + f'cohort_scs_log_decoded_{arch}dim.nc')


