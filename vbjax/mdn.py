import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp
import optax
from tqdm import trange

# For SBI comparison
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

# For plotting
import matplotlib.pyplot as plt

# --- Existing JAX MDN Code ---

def init_dense_net_params(layer_sizes, key):
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for in_size, out_size, key in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        w_key, b_key = jax.random.split(key)
        w = jax.random.glorot_normal(w_key, (in_size, out_size))
        b = jnp.zeros(out_size)
        params.append((w, b))
    return params

def dense_net_forward(params, x):
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jax.nn.relu(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits

def get_mdn_params(net_output, num_dimensions, num_components):
    D, K = num_dimensions, num_components
    num_chol_off_diag = D * (D - 1) // 2
    
    output_parts = jnp.split(net_output, [K, K + K * D, K + K * D + K * D])
    logits, mus_flat, chol_diag_flat = output_parts[:3]
    chol_off_diag_flat = output_parts[3] if len(output_parts) > 3 else jnp.array([])

    pis = jax.nn.softmax(logits)
    mus = mus_flat.reshape(K, D)
    chol_diag = jnp.exp(chol_diag_flat.reshape(K, D))
    
    if num_chol_off_diag > 0:
        chol_off_diag = chol_off_diag_flat.reshape(K, num_chol_off_diag)
    else: # Handle 1D case where there are no off-diagonal elements
        chol_off_diag = jnp.empty((K, 0))


    def build_covariance(diag_elements, off_diag_elements):
        L = jnp.zeros((D, D))
        L = L.at[jnp.diag_indices(D)].set(diag_elements)
        if D > 1:
            L = L.at[jnp.tril_indices(D, k=-1)].set(off_diag_elements)
        return L @ L.T

    covariances = jax.vmap(build_covariance)(chol_diag, chol_off_diag)
    return pis, mus, covariances

def mdn_log_prob(pis, mus, covariances, theta):
    log_probs_per_component = jax.vmap(multivariate_normal.logpdf, in_axes=(None, 0, 0))(
        theta, mus, covariances
    )
    log_pis = jnp.log(pis)
    return logsumexp(log_pis + log_probs_per_component)

def network_log_prob(net_params, x, theta, num_dimensions, num_components):
    net_output = dense_net_forward(net_params, x)
    pis, mus, covariances = get_mdn_params(net_output, num_dimensions, num_components)
    return mdn_log_prob(pis, mus, covariances, theta)

# --- Test Harness ---

def simulator(theta):
    """A simple simulator: theta -> x."""
    noise = jnp.randn(theta.shape) * 0.1
    x = theta**2 + noise
    return x

def sbi_simulator(theta):
    """Wrapper for SBI compatibility (uses torch)."""
    theta_np = theta.numpy()
    x_np = theta_np**2 + jnp.array(jnp.randn(*theta_np.shape) * 0.1)
    return torch.from_numpy(x_np).float()

def main():
    # --- Configuration ---
    theta_dim = 2
    x_dim = 2
    num_components = 5
    num_simulations = 2000
    learning_rate = 1e-3
    num_epochs = 200
    batch_size = 128
    
    key = jax.random.PRNGKey(42)

    # --- Generate Data ---
    prior_min = -2.0 * jnp.ones(theta_dim)
    prior_max = 2.0 * jnp.ones(theta_dim)

    key, subkey = jax.random.split(key)
    thetas = jax.random.uniform(subkey, (num_simulations, theta_dim), minval=prior_min, maxval=prior_max)
    xs = simulator(thetas)

    # --- JAX MDN Training ---
    print("Training JAX MDN...")
    
    num_chol_off_diag = theta_dim * (theta_dim - 1) // 2
    mdn_output_size = num_components + (num_components * theta_dim) + \
                      (num_components * theta_dim) + (num_components * num_chol_off_diag)

    layer_sizes = [x_dim, 64, 64, mdn_output_size]
    key, subkey = jax.random.split(key)
    net_params = init_dense_net_params(layer_sizes, subkey)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(net_params)
    
    vmapped_log_prob = jax.vmap(network_log_prob, in_axes=(None, 0, 0, None, None))

    def loss_fn(params, x_batch, theta_batch):
        return -jnp.mean(vmapped_log_prob(params, x_batch, theta_batch, theta_dim, num_components))

    @jax.jit
    def train_step(params, opt_state, x_batch, theta_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, theta_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    num_batches = num_simulations // batch_size
    for epoch in trange(num_epochs):
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, num_simulations)
        xs_perm, thetas_perm = xs[perms, :], thetas[perms, :]
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            net_params, opt_state, loss = train_step(
                net_params, opt_state, xs_perm[start:end], thetas_perm[start:end]
            )

    print("JAX training complete.")

    # --- SBI Training ---
    print("\nTraining with sbi...")
    prior = BoxUniform(low=torch.tensor(prior_min), high=torch.tensor(prior_max))
    
    inference = SNPE(prior=prior)
    
    # sbi uses torch tensors
    thetas_torch = torch.from_numpy(jnp.asarray(thetas)).float()
    xs_torch = torch.from_numpy(jnp.asarray(xs)).float()

    inference = inference.append_simulations(thetas_torch, xs_torch)
    density_estimator = inference.train()
    posterior_sbi = inference.build_posterior(density_estimator)
    print("SBI training complete.")

    # --- Comparison ---
    x_obs = jnp.array([0.5, 0.5])
    x_obs_torch = torch.from_numpy(jnp.asarray(x_obs)).float()
    
    # JAX Posterior visualization
    grid_res = 100
    theta1_vals = jnp.linspace(prior_min[0], prior_max[0], grid_res)
    theta2_vals = jnp.linspace(prior_min[1], prior_max[1], grid_res)
    grid_thetas = jnp.array(jnp.meshgrid(theta1_vals, theta2_vals)).T.reshape(-1, 2)
    
    net_output = dense_net_forward(net_params, x_obs)
    pis, mus, covs = get_mdn_params(net_output, theta_dim, num_components)
    
    log_probs_jax = jax.vmap(mdn_log_prob, in_axes=(None, None, None, 0))(pis, mus, covs, grid_thetas)
    prob_grid_jax = jnp.exp(log_probs_jax).reshape(grid_res, grid_res)

    # SBI Posterior sampling
    samples_sbi = posterior_sbi.sample((5000,), x=x_obs_torch).numpy()

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Posterior Comparison for Observation x = {x_obs.tolist()}", fontsize=16)

    # JAX plot
    axes[0].contourf(theta1_vals, theta2_vals, prob_grid_jax.T, levels=20, cmap='viridis')
    axes[0].set_title("JAX Implementation")
    axes[0].set_xlabel("$\\theta_1$")
    axes[0].set_ylabel("$\\theta_2$")
    axes[0].set_aspect('equal')

    # SBI plot
    axes[1].hist2d(samples_sbi[:, 0], samples_sbi[:, 1], bins=50, density=True, cmap='viridis')
    axes[1].set_title("SBI Implementation")
    axes[1].set_xlabel("$\\theta_1$")
    axes[1].set_ylabel("$\\theta_2$")
    axes[1].set_xlim(prior_min[0], prior_max[0])
    axes[1].set_ylim(prior_min[1], prior_max[1])
    axes[1].set_aspect('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()

