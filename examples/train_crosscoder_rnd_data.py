"""
CrossCoder example: train a linear auto-encoder over a cohort of
connectomes following the vxc_train workflow.

Swap ``load_cohort`` for ``xarray.load_dataarray`` (and the flat-triu
extraction below) to run on real connectomes.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import vbjax as vb


def load_cohort(n_subjects=200, n_nodes=64, seed=0):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_subjects, 5))
    loadings = rng.normal(size=(5, n_nodes, n_nodes))
    mats = np.einsum('sk,knm->snm', factors, loadings)
    mats = np.maximum(0.5 * (mats + mats.transpose(0, 2, 1)), 0.0)
    return mats / mats.max()


def main():
    mats = load_cohort()
    n_sub, n_nodes, _ = mats.shape
    i, j = np.triu_indices(n_nodes, k=1)
    sc_flat = mats[:, i, j]

    variational = False
    normalize = 'zscore' if variational else 'center'

    xc = vb.CrossCoder(variational=variational)
    xc.add_view(sc_flat, 'SC', normalize=normalize, nonneg=True)
    xc.tts = int(n_sub * 0.8)
    print(f"Train/test split: {xc.tts} / {n_sub - xc.tts}")

    if variational:
        dims     = [2, 3, 4]
        niter    = 25000
        lr       = 8e-4
        beta_end = 1e-6
        anneal   = 2200
    else:
        dims     = [2, 3, 4]
        niter    = 30000
        lr       = 5e-5
        beta_end = 0.0
        anneal   = 0

    results = []
    for dim in dims:
        print(f"\n{'='*40}  Dim {dim}  {'='*40}")
        trace, _, cr = xc.train(
            nlat=dim, niter=niter, lr=lr, mb=64,
            beta_start=0.0, beta_end=beta_end, anneal_steps=anneal,
        )
        trace = np.array(trace)
        mu = np.array(xc.encode(dim, 'SC', sample=False))
        std_z = mu.std(axis=0)

        if variational:
            final_r  = float(trace[-1, 2])
            final_kl = float(trace[-1, 3])
            print(f"CR: {cr:.4f}  |  R: {final_r:.4f}  |  KL: {final_kl:.2f}"
                  f"  |  Latent std: [{std_z.min():.3f}, {std_z.max():.3f}]")
            mse = final_r
        else:
            print(f"CR: {cr:.4f}  |  Latent std: [{std_z.min():.3f}, {std_z.max():.3f}]")
            mse = float(np.exp(trace[-1, 1]))

        results.append({
            'dim': dim, 'cr': cr, 'mse': mse,
            'trace': trace,
            'log_freq': xc.history[-1]['log_freq'],
        })

    # xc.to_pkl('xc_example.pkl')


if __name__ == '__main__':
    main()
