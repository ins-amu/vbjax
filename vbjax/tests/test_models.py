import numpy as np
import jax.numpy as jp
import vbjax as vb


def true_dopa():

    a=0.04
    b=5.
    c=140.
    ga=12.
    gg=12.
    Delta=1.
    Eta=18.
    Iext=0.
    Ea=0.
    Eg=-80.
    tauSa=5.
    tauSg=5.
    Sja=0.8
    Sjg=1.2
    ud=12.
    alpha=0.013
    beta=.4
    k=10e4 #10e4,
    Vmax=1300.
    Km=150.
    Bd=0.2
    Ad=1.
    tau_Dp=500.
    params=np.array([a, b, c, ga, gg, Eta, Delta, Iext, Ea, Eg, Sja, Sjg, tauSa, tauSg, alpha, beta, ud, k, Vmax, Km, Bd, Ad, tau_Dp])

    n_nodes = 8
    conn_inhibitor, conn_excitator, conn_dopamine = np.random.randn(3, n_nodes, n_nodes)**2

    dt = 0.01
    t0 = 0.0
    tf = 1.0
    ckk= 1e-4                       #coupling scaling
    sigma=1e-3                      #amplitude of noise - for sigma=0 --> Heun methd original                 
    r0 = np.full(n_nodes, 0.1)
    V0 = np.full(n_nodes, -70.0)
    u0 = np.full(n_nodes, 0.0)  
    Sa0 = np.full(n_nodes, 0.0)  
    Sg0 = np.full(n_nodes, 0.0) 
    Dp0 = np.full(n_nodes, 0.05)  
    y0 = np.concatenate((r0, V0, u0, Sa0, Sg0, Dp0))

    def aQIFdopa(y,t,params,coupling_inhibitor,coupling_excitator,coupling_dopamine): 
        r = y[0*n_nodes : 1*n_nodes]
        V = y[1*n_nodes : 2*n_nodes]
        u = y[2*n_nodes : 3*n_nodes]
        Sa = y[3*n_nodes : 4*n_nodes]
        Sg = y[4*n_nodes : 5*n_nodes]
        Dp = y[5*n_nodes : 6*n_nodes]
        a, b, c, ga, gg, Eta, Delta, Iext, Ea, Eg, Sja, Sjg, tauSa, tauSg, alpha, beta, ud, k, Vmax, Km, Bd, Ad, tau_Dp=params
        c_inh = coupling_inhibitor
        c_exc = coupling_excitator
        c_dopa = coupling_dopamine

        dydt = np.concatenate((
            2. * a * r * V + b * r - ga * Sa * r - gg * Sg * r + (a * Delta) / np.pi,
            a * V**2 + b * V + c + Eta - (np.pi**2 * r**2) / a + (Ad * Dp + Bd) * ga * Sa * (Ea - V) + gg * Sg * (Eg - V) + Iext - u,
            alpha * (beta * V - u) + ud * r,
            -Sa / tauSa + Sja * c_exc,
            -Sg / tauSg + Sjg * c_inh,
            (k * c_dopa - Vmax * Dp / (Km + Dp)) / tau_Dp
        )).flatten()

        return dydt

    def network(y, t, ckk, params):
        r = y[0*n_nodes : 1*n_nodes]

        aff_inhibitor = conn_inhibitor @ r * ckk
        aff_excitator = conn_excitator @ r * ckk
        aff_dopamine = conn_dopamine @ r * ckk

        dx = aQIFdopa(y, t, params, aff_inhibitor, aff_excitator, aff_dopamine)
        return dx

    def heun_SDE(network,y0,t0,t_max,dt,params,ckk,sigma):
        num_steps = int((t_max - t0) / dt)
        y_all = np.empty((num_steps, len(y0)))
        t_all = np.empty((num_steps, ))
        stochastic_matrix = np.random.normal(0, 1, (len(y0),num_steps))
        t=t0;  i=0
        t_all[i] = t0
        y_all[i, :] = y0
        y=y0
        for step in range(num_steps):
            dw = stochastic_matrix[:,step]*sigma * np.sqrt(dt)
            dy1 = network(y, t, ckk,params)
            ye = y + dt * dy1 + dw  
            y = y + 0.5 * dt * (dy1 + network(ye, t + dt, ckk,params)) + dw
            t=t+dt
            t_all[i]=t
            y_all[i,:]=y
            i+=1
        return y_all, t_all, stochastic_matrix.T
    
    y1, t1, dw = heun_SDE(network,y0,t0,tf,dt,params,ckk,sigma)
    return y1, t1, dw, ckk, params, conn_inhibitor, conn_excitator, conn_dopamine, n_nodes, r0, V0, u0, Sa0, Sg0, Dp0, network, dt, sigma

def test_dopa():

    y1, t1, dw, ckk, params, conn_inhibitor, conn_excitator, conn_dopamine, n_nodes, r0, V0, u0, Sa0, Sg0, Dp0, network, dt, sigma = true_dopa()

    _, loop = vb.make_sde(dt=dt, dfun=vb.dopa_net_dfun, gfun=sigma)

    j_y0 = jp.array([r0, V0, u0, Sa0, Sg0, Dp0])
    j_params = vb.DopaTheta(*params, wi=ckk, we=ckk, wd=ckk, sigma=sigma)
    j_Ci, j_Ce, j_Cd = [jp.array(_) for _ in (conn_inhibitor, conn_excitator, conn_dopamine)]
    j_dw = jp.array(dw).reshape(-1, 6, n_nodes)
    assert j_dw.shape == (t1.size, 6, n_nodes)


    j_y2 = loop(j_y0, j_dw, (j_Ci, j_Ce, j_Cd, j_params))
    
    # compare derivatives
    for i in range(t1.size):
        dy1 = network(y1[i], t1[i], ckk, params).reshape((6, -1))
        dy2 = vb.dopa_net_dfun(y1[i].reshape((6,-1)), (j_Ci, j_Ce, j_Cd, j_params))
        for j in range(6):
            np.testing.assert_allclose(dy1[j], dy2[j], rtol=1e-5, atol=1e-5)

    # compare trajectories
    y1_ = y1.reshape((-1, 6, n_nodes))
    if False:
        # do plots 
        import matplotlib.pyplot as pl
        for i in range(6):
            pl.subplot(3, 2, i + 1)
            pl.plot(t1, y1_[:,i], 'k', alpha=0.2)
            pl.plot(t1, j_y2[i], 'r', alpha=0.2)
            pl.grid(1)
            np.testing.assert_allclose(y1_[:,i], j_y2[i])
        pl.savefig('dopa.png', dpi=300)
    else:
        # don't bother plots just assert all close each var
        for i in range(6):
            np.testing.assert_allclose(y1_[:,i], j_y2[:,i], rtol=1e-5, atol=1e-5)
    