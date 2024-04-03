import numpy as np
import matplotlib.pyplot as pl
import jax.numpy as jp
import vbjax as vb
from vbjax.app.dopa import sweep_node, sweep_network

# start with default parameters        
params = vb.dopa_default_theta

# update params and sweep over Km and Vmax
params = params._replace(
    Bd=jp.r_[0.2, 0.5, 0.8, 1.0]
    )

# initial conditions
y0 = jp.array([0.25, -50.0, 0.0, 0.4, 0.05, 0.0])

# run sweep
end_time = 30.0

# svar names
svars = 'r V u Sa Sg Dp'.split(' ')

# choose noise per state var; for this scenario
# put noise in slow variables for "spontaneous" bursts in r,V
sigma = jp.r_[0, 0, 0, 1e-1, 1e-1, 1e-1]

# run for small dt
pl.figure(figsize=(10, 8))

for dt in [0.001,]:
    pkeys, ys = sweep_node(y0, params, dt=dt, sigma=sigma, T=end_time, cores=4)
    ys.block_until_ready()
    # plot some of it
    t = np.r_[:end_time:1j*ys.shape[-2]]
    for i_svar in range(6):
        pl.subplot(3, 2, i_svar + 1)
        kw = {}
        if i_svar == 2:
            kw['label'] = [f'Bd={bd:0.2f}' for bd in params.Bd]
        pl.plot(t, ys[:, :, i_svar].T, **kw)
        pl.xlabel('time (ms)')
        pl.ylabel(f"{svars[i_svar]}(t)")
        pl.grid(1)

pl.subplot(3,2,3)
pl.legend()
pl.tight_layout()
pl.show()
