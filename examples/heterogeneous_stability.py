"""
**Example**: Stability analysis of a heterogeneous network.

This example extends heterogeneous_network.py to demonstrate how to
analyze the stability of a network's fixed point by computing the
eigenvalues of the Jacobian matrix.

"""
import os
import jax
import jax.numpy as jp
from jax.tree_util import tree_flatten, tree_unflatten
import vbjax as vb
import collections
import matplotlib.pyplot as plt

# Create a directory for saving images
os.makedirs('images', exist_ok=True)

# 1. Define the network structure (same as heterogeneous_network.py)
n_jr_nodes = 4
n_mpr_nodes = 2
n_total_nodes = n_jr_nodes + n_mpr_nodes

connectivity = jp.zeros((n_total_nodes, n_total_nodes))
connectivity = connectivity.at[:n_jr_nodes, :n_jr_nodes].set(0.2)
connectivity = connectivity.at[n_jr_nodes:, n_jr_nodes:].set(0.2)
connectivity = connectivity.at[n_jr_nodes:, :n_jr_nodes].set(0.1)
connectivity = connectivity.at[jp.diag_indices(n_total_nodes)].set(0)

# 2. Define PyTrees for the heterogeneous network
HeteroState = collections.namedtuple('HeteroState', ['jr', 'mpr'])
HeteroTheta = collections.namedtuple('HeteroTheta', ['jr', 'mpr'])

# 3. Define the network dynamics function (dfun)
def hetero_dfun(state, p, coupling_strength):
    jr_state, mpr_state = state
    jr_p, mpr_p = p

    jr_r = jr_state.y0
    mpr_r = mpr_state.r
    all_r = jp.concatenate([jr_r, mpr_r])

    coupling_input = coupling_strength * connectivity @ all_r
    jr_coupling_input = coupling_input[:n_jr_nodes]
    mpr_coupling_input = (coupling_input[n_jr_nodes:], 0.0)

    d_jr_array = vb.jr_dfun(jr_state, jr_coupling_input, jr_p)
    d_mpr_array = vb.mpr_dfun(mpr_state, mpr_coupling_input, mpr_p)

    d_jr = vb.JRState(*[d_jr_array[i] for i in range(len(vb.JRState._fields))])
    d_mpr = vb.MPRState(*[d_mpr_array[i] for i in range(len(vb.MPRState._fields))])

    return HeteroState(jr=d_jr, mpr=d_mpr)

# 4. Set up simulation parameters
dt = 0.1
coupling_strength = 0.05

# Initial states and parameters
jr_state_init = vb.JRState(*[jp.ones(n_jr_nodes) * v for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])
mpr_state_init = vb.MPRState(r=jp.ones(n_mpr_nodes) * 0.1, V=jp.ones(n_mpr_nodes) * -2.0)
initial_state = HeteroState(jr=jr_state_init, mpr=mpr_state_init)
params = HeteroTheta(jr=vb.jr_default_theta, mpr=vb.mpr_default_theta)

# Define a deterministic dfun for finding the fixed point
dfun = lambda state, p: hetero_dfun(state, p, coupling_strength)

# 5. Find a fixed point of the system
# We'll do this by running a deterministic simulation until it settles.
# This gives an approximation of a stable fixed point.
print("Finding fixed point...")
run_sim = vb.make_ode(dt, dfun)
# Run for a bit to let transients pass away
fp_state = run_sim(initial_state, params, time=1000.0)
print("Fixed point found.")

# 6. Linearize the system and find eigenvalues
# To use jax.jacobian, we need to work with flat vectors, not PyTrees.
# We'll create functions to flatten the state PyTree and wrap the dfun.

# Get the PyTree structure definition from our initial state
_, state_tree_def = tree_flatten(initial_state)

def flatten_state(state_pytree):
    "Flattens a state PyTree into a single 1D array."
    leaves, _ = tree_flatten(state_pytree)
    return jp.concatenate([leaf.ravel() for leaf in leaves])

def unflatten_state(flat_state_vec, tree_def):
    "Unflattens a 1D array back into a state PyTree."
    # Get the shapes of the leaves from a template PyTree
    template_leaves, _ = tree_flatten(tree_def.unflatten([0]*tree_def.num_leaves))
    leaf_shapes = [leaf.shape for leaf in template_leaves]
    
    # Unflatten the vector
    leaf_sizes = [jp.prod(jp.array(s)) for s in leaf_shapes]
    indices = jp.cumsum(jp.array([0] + leaf_sizes))
    leaves = [
        flat_state_vec[indices[i]:indices[i+1]].reshape(leaf_shapes[i])
        for i in range(len(leaf_shapes))
    ]
    return tree_unflatten(tree_def, leaves)

def flat_dfun(flat_state_vec, p):
    "A wrapper for dfun that works with flattened state vectors."
    state = unflatten_state(flat_state_vec, state_tree_def)
    d_state_pytree = dfun(state, p)
    return flatten_state(d_state_pytree)

# Now we can compute the Jacobian at the fixed point
print("Computing Jacobian and eigenvalues...")
fp_flat = flatten_state(fp_state)
jacobian_matrix = jax.jacobian(flat_dfun)(fp_flat, params)
eigenvalues = jp.linalg.eigvals(jacobian_matrix)
print("Done.")

# 7. Plot the eigenvalues in the complex plane
print("Plotting results...")
plt.figure(figsize=(8, 8))
plt.scatter(jp.real(eigenvalues), jp.imag(eigenvalues), c='b', marker='o')
plt.axvline(0, color='r', linestyle='--', label='Stability Boundary (Re=0)')
plt.title(f'Eigenvalues of the Jacobian (g={coupling_strength})')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('images/heterogeneous_network_eigenvalues.png')
plt.show()

print("Plot saved to images/heterogeneous_network_eigenvalues.png")

# Interpretation:
# If all eigenvalues are in the left half-plane (real part < 0), the
# fixed point is stable. If any eigenvalue has a positive real part,
# the fixed point is unstable. By running this analysis for different
# values of `coupling_strength`, you can find bifurcations where the
# system's stability changes.
