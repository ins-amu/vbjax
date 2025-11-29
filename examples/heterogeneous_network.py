"""
**Example**: A network with multiple neural mass models.

This example demonstrates how to simulate a network with different neural mass
models assigned to different nodes. We'll create a network with some nodes
using the Jansen-Rit (JR) model and others using the Montbrio-Pazo-Roxin (MPR)
model.

"""
import os
import jax.numpy as jp
import vbjax as vb
import collections
import matplotlib.pyplot as plt

# Create a directory for saving images
os.makedirs('images', exist_ok=True)

# 1. Define the network structure
n_jr_nodes = 4
n_mpr_nodes = 2
n_total_nodes = n_jr_nodes + n_mpr_nodes

# Define a connectome: JR nodes are reciprocally connected, MPR nodes too,
# and there are connections from JR to MPR nodes.
connectivity = jp.zeros((n_total_nodes, n_total_nodes))
connectivity = connectivity.at[:n_jr_nodes, :n_jr_nodes].set(0.2)
connectivity = connectivity.at[n_jr_nodes:, n_jr_nodes:].set(0.2)
connectivity = connectivity.at[n_jr_nodes:, :n_jr_nodes].set(0.1)
connectivity = connectivity.at[jp.diag_indices(n_total_nodes)].set(0)


# 2. Define PyTrees for the heterogeneous network
# We use namedtuples to structure the state, parameters, and noise for the
# different model types. This is a flexible way to handle heterogeneous
# networks and is compatible with JAX's `tree_map` and `jit`.
HeteroState = collections.namedtuple('HeteroState', ['jr', 'mpr'])
HeteroTheta = collections.namedtuple('HeteroTheta', ['jr', 'mpr'])


# 3. Define the network dynamics function (dfun)
# This function computes the time derivatives for each node in the network.
def hetero_dfun(state, p, coupling_strength):
    jr_state, mpr_state = state
    jr_p, mpr_p = p

    # Extract firing rates from each model type
    # For JR, the firing rate is proportional to the pyramidal cell activity (y0).
    # For MPR, the firing rate is the `r` variable.
    jr_r = jr_state.y0
    mpr_r = mpr_state.r

    # Combine firing rates from all nodes into a single vector
    all_r = jp.concatenate([jr_r, mpr_r])

    # Calculate the coupling input to each node
    coupling_input = coupling_strength * connectivity @ all_r

    # Distribute the coupling input to the respective model types
    jr_coupling_input = coupling_input[:n_jr_nodes]
    # MPR's dfun expects a tuple for coupling (e.g., for different receptors)
    mpr_coupling_input = (coupling_input[n_jr_nodes:], 0.0)

    # Calculate the derivatives for each model type
    d_jr_array = vb.jr_dfun(jr_state, jr_coupling_input, jr_p)
    d_mpr_array = vb.mpr_dfun(mpr_state, mpr_coupling_input, mpr_p)

    # The dfun for each model returns a flat array. We need to repack it
    # into the same namedtuple structure as the state.
    d_jr = vb.JRState(*[d_jr_array[i] for i in range(len(vb.JRState._fields))])
    d_mpr = vb.MPRState(*[d_mpr_array[i] for i in range(len(vb.MPRState._fields))])

    return HeteroState(jr=d_jr, mpr=d_mpr)


# 4. Set up the simulation
dt = 0.1
duration = 1000.0
t_steps = int(duration / dt)
coupling_strength = 0.05
noise_level = 0.01

# Wrap the dfun to include the coupling strength parameter
dfun = lambda state, p: hetero_dfun(state, p, coupling_strength)

# Create the SDE loop
_, loop = vb.make_sde(dt=dt, dfun=dfun, gfun=noise_level)

# 5. Set up initial conditions, parameters, and noise
# Initial states
jr_state_init = vb.JRState(*[jp.ones(n_jr_nodes) * v for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])
mpr_state_init = vb.MPRState(r=jp.ones(n_mpr_nodes) * 0.1, V=jp.ones(n_mpr_nodes) * -2.0)
initial_state = HeteroState(jr=jr_state_init, mpr=mpr_state_init)

# Parameters
params = HeteroTheta(jr=vb.jr_default_theta, mpr=vb.mpr_default_theta)

# Noise: the noise PyTree must have the same structure as the state
jr_noise = vb.JRState(*[vb.randn(t_steps, n_jr_nodes) for _ in range(len(vb.JRState._fields))])
mpr_noise = vb.MPRState(*[vb.randn(t_steps, n_mpr_nodes) for _ in range(len(vb.MPRState._fields))])
noise = HeteroState(jr=jr_noise, mpr=mpr_noise)


# 6. Run the simulation
print("Running simulation...")
result = loop(initial_state, noise, params)
print("Simulation finished.")


# 7. Plot the results
print("Plotting results...")
time = jp.arange(t_steps) * dt

plt.figure(figsize=(12, 6))

# Plot JR model states (y0: pyramidal cell activity)
plt.subplot(2, 1, 1)
plt.plot(time, result.jr.y0, alpha=0.8)
plt.title('Jansen-Rit Model (Pyramidal Cell Activity)')
plt.xlabel('Time (ms)')
plt.ylabel('Activity')
plt.grid(True)

# Plot MPR model states (r: firing rate)
plt.subplot(2, 1, 2)
plt.plot(time, result.mpr.r, alpha=0.8)
plt.title('Montbrio-Pazo-Roxin Model (Firing Rate)')
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate')
plt.grid(True)

plt.tight_layout()
plt.savefig('images/heterogeneous_network_activity.png')
plt.show()
print("Plot saved to images/heterogeneous_network_activity.png")
