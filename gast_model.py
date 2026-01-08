import jax.numpy as jp
import collections  

# Model implementation

Theta = collections.namedtuple(
    typename='Theta',
    field_names='C, k, Delta, v_r, v_theta, g, E, I, tau_u, b, kappa, tau_s, J, Js, we,')

default_theta = Theta(
    C=100.0, k=0.7, Delta=0.5, v_r=-60.0, v_theta=-40.0, g=1.0, E=0.0, I=40.0, 
    tau_u=33.33, b=-2.0, kappa=100.0, tau_s=6.0, Js=.0, J=15.0, we=0.0, 
)

def dfun(y, cy, p: Theta):
    "Adaptive QIF model with dopamine modulation."

    r, v, u, s = y
    c_exc = cy
    C, k, Delta, v_r, v_theta, g, E, I, tau_u, b, kappa, tau_s, J, Js, *_ = p

    dr = ((Delta * k**2 * (v - v_r)) / (jp.pi * C) + r * (k * (2 * v - v_r - v_theta) - g*s)) / C
    dv = (k * v * (v - v_r - v_theta) - jp.pi * C * r * (Delta + jp.pi * C * r/ k) + k * v_r * v_theta - u + I + g * s * (E - v)) / C
    du = (b * (v - v_r) -u) / tau_u + kappa * r
    ds = - s / tau_s + Js * c_exc + J * r

    return jp.array([dr, dv, du, ds])

def net(y, p):
    "Canonical form for network of dopa nodes."
    Ce, node_params = p
    r = y[0]
    c_exc = node_params.we * Ce @ r
  
    return dfun(y, (c_exc), node_params)

def stay_positive(y, _):
    # at, set are JAX function used for immutable updates to an array
    # if where<0 is true, set the value to 0, conversely it leaves the original value 
    # in this way r, is never negative
    y = y.at[0].set( jp.where(y[0]<0, 0, y[0]) ) #r

    return y

# Model implementation with dopamine

dopa_Theta = collections.namedtuple(
    typename='dopa_Theta',
    field_names='C, k, Delta, v_r, v_theta, Bd, ga, gg, Ea, Eg, I, tau_u, b, kappa, tau_sa, tau_sg, Ja, Jg, Jsa, Jsg, Jdopa, Vmax, Km, tau_Dp, Rd, Sd, Z, tau_Md, we, wi, wd, sigma_V, sigma_u')

dopa_default_theta = dopa_Theta(
    C=100.0, k=0.7, Delta=0.5, v_r=-60.0, v_theta=-40.0, Bd=1., ga=1.0, gg=1., Ea=0.0, Eg=-80., I=46.5, 
    tau_u=33.33, b=-2.0, kappa=100.0, tau_sa=6.0, tau_sg=6., Ja=13., Jg=0., Jsa=13., Jsg=15., 
    Jdopa=100000.0, Vmax=1300.0, Km=150.0, Rd=1., Sd=-10.0, Z=.5, tau_Dp=500.0, tau_Md=1000.0, 
    we=1e-2, wi=1e-2, wd=1e-2, sigma_V=0.1, sigma_u=0.01,
)

def dopa_dfun(y, cy, p: dopa_Theta):
    "Adaptive QIF model with dopamine modulation."

    r, v, u, sa, sg, Dp, Md = y
    c_exc, c_inh, c_dopa = cy
    C, k, Delta, v_r, v_theta, Bd, ga, gg, Ea, Eg, I, tau_u, b, kappa, tau_sa, tau_sg, Ja, Jg, Jsa, Jsg, Jdopa, Vmax, Km, tau_Dp, Rd, Sd, Z, tau_Md, *_ = p

    dr = ((Delta * k**2 * (v - v_r)) / (jp.pi * C) + r * (k * (2 * v - v_r - v_theta) - (Bd + Md) * ga * sa - gg * sg)) / C
    dv = (k * v * (v - v_r - v_theta) - jp.pi * C * r * (Delta + jp.pi * C * r/ k) + k * v_r * v_theta - u + I + (Bd + Md) * ga * sa * (Ea - v) + gg * sg * (Eg - v)) / C
    du = (b * (v - v_r) -u) / tau_u + kappa * r
    dsa = - sa / tau_sa + Jsa * c_exc + Ja * r
    dsg = - sg / tau_sg + Jsg * c_inh + Jg * r
    dDp = (Jdopa * c_dopa - Vmax * Dp / (Km + Dp)) / tau_Dp
    dMd = (-Md + Rd / (1 + jp.exp(Sd * jp.log((Dp+Z))))) / tau_Md

    return jp.array([dr, dv, du, dsa, dsg, dDp, dMd])

def dopa_net(y, p):
    "Canonical form for network of dopa nodes."
    Ce, Ci, Cd, node_params = p
    r = y[0]
    c_exc = node_params.we * Ce @ r
    c_inh = node_params.wi * Ci @ r
    c_dopa = node_params.wd * Cd @ r

    return dopa_dfun(y, (c_exc, c_inh, c_dopa), node_params)

def dopa_stay_positive(y, _):
    # at, set are JAX function used for immutable updates to an array
    # if where<0 is true, set the value to 0, conversely it leaves the original value 
    # in this way r, Dp and Md are never negative
    y = y.at[0].set( jp.where(y[0]<0, 0, y[0]) ) #r
    y = y.at[5].set( jp.where(y[5]<0, 0, y[5]) ) #Dp
    y = y.at[6].set( jp.where(y[6]<0, 0, y[6]) ) #Md

    return y