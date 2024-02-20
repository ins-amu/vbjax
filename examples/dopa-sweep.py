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
y0 = jp.array([0., -2.0, 0.0, 0.0, 0.0, 0.0])

# run sweep
end_time = 256.0
pkeys, ys = sweep_node(y0, params, T=end_time, cores=4)

# pkeys provides the names for the extra dims of ys result
print(pkeys, ys.shape)

# now similar for network sweep
n_nodes = 8
Ci, Ce, Cd = jp.zeros((3, n_nodes))
pkeys, ys = sweep_network(y0, params, Ci, Ce, Cd, T=end_time, cores=4)
print(pkeys, ys.shape)
