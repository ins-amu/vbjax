import jax.numpy as jp
import vbjax as vb
from vbjax.app.dopa import sweep_node, sweep_network

# start with default parameters        
params = vb.dopa_default_theta

# update params and sweep over Km and Vmax
params = params._replace(
    Eta=18.2,
    Km=jp.r_[100:200:32j],
    Vmax=jp.r_[1000:2000:32j],
    )

# initial conditions
y0 = jp.array([0.1, -60.0, 0.0, 0.0, 0.0, 0.0])

# run sweep
end_time = 256.0
pkeys, ys = sweep_node(y0, params, T=end_time, cores=4)

# for large sweeps, ys may not yet be filled with outputs
ys.block_until_ready()

# pkeys provides the names for the extra dims of ys result
print(pkeys, ys.shape)

# now similar for network sweep
n_nodes = 8
Ci, Ce, Cd = vb.randn(3, n_nodes, n_nodes)
pkeys, ys = sweep_network(y0, params, Ci, Ce, Cd, T=end_time, cores=4)
ys.block_until_ready()
print(pkeys, ys.shape)

# plot some of it
import matplotlib.pyplot as pl

pl.plot(ys[0,0,:,0], 'k')
pl.show()
