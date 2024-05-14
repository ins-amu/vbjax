import matplotlib.pyplot as pl
import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb


for dt in [0.01, 0.001]:

    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda x,p: vb.dopa_dfun(x, (0,0,0), p),
        gfun=1e-9,
        adhoc=vb.dopa_r_positive,
        return_euler=True
    )

    y0 = jp.r_[0.25, -50.0, 0.0, 0.33, 0.02, 0.0]
    total_time = 5.0
    dW = vb.randn( int(total_time/dt), 6 )
    eys, ys = loop(y0, dW, vb.dopa_default_theta)
    t = jp.r_[:dW.shape[0]]*dt

    atol = 1e-3
    rtol = 1e-3
    svmax = ys.max(axis=0)*1.1
    # can we try to guess step size?
    abs_err = jp.abs(eys - ys)
    total_tol = atol + rtol * jp.abs(ys)
    # jp.allclose(eys, ys, rtol=rtol, atol=atol), but per sample
    ok = abs_err <= total_tol
    # lamba eq 2.2, dt==h, but approx per svar
    theta = 0.8
    hmax = 0.1
    p = 2 # Heun is 2nd order
    rho = 0 # error-per-step (rho=1 error per unit step)
    # ideal dt
    hp = theta*dt*(total_tol/abs_err)**(1/(p - rho))

    for i in range(6):
        pl.subplot(4, 3, [1,2,3,7,8,9][i])
        pl.plot(t, eys[:, i], 'r', alpha=0.5)
        pl.plot(t, ys[:, i], 'k', alpha=0.5)
        pl.plot(t[~ok[:,i]], np.ones((~ok[:,i]).sum())*svmax[i], 'rx', alpha=0.5)
        pl.subplot(4, 3, [1,2,3,7,8,9][i]+3)
        pl.semilogy(t, abs_err[:, i], 'r', alpha=0.5)
        pl.semilogy(t, total_tol[:, i], 'k', alpha=0.5)
        pl.semilogy(t, hp[:, i], 'g', alpha=0.5)
        pl.axhline(dt, color='g')

pl.savefig(__file__ + '.jpg', dpi=300)
pl.show()
