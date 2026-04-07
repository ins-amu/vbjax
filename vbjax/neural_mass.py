import collections
import jax.numpy as np


JRTheta = collections.namedtuple(
    typename='JRTheta',
    field_names='A B a b v0 nu_max r J a_1 a_2 a_3 a_4 mu I'.split(' '))

jr_default_theta = JRTheta(
    A=3.25, B=22.0, a=0.1, b=0.05, v0=5.52, nu_max=0.0025, 
    r=0.56, J=135.0, a_1=1.0, a_2=0.8, a_3=0.25, a_4=0.25, mu=0.22, I=0.0)

JRState = collections.namedtuple(
    typename='JRState',
    field_names='y0 y1 y2 y3 y4 y5'.split(' '))
    
def jr_dfun(ys, c, p):
    r"""
    Jansen-Rit mass model.

    Parameters
    ----------
    ys : array
        State vector of shape (6, ...).
        The components are:
        - y0, y3: Excitatory Postsynaptic Potential (EPSP) of pyramidal population and its derivative.
        - y1, y4: EPSP of excitatory interneurons and its derivative.
        - y2, y5: Inhibitory Postsynaptic Potential (IPSP) of inhibitory interneurons and its derivative.
    c : array
        Coupling input.
    p : JRTheta
        Parameters.

    Returns
    -------
    dys : array
        Derivatives of the state vector.

    Notes
    -----
    The equations are:

    .. math::
        \dot{y}_0 &= y_3 \\
        \dot{y}_1 &= y_4 \\
        \dot{y}_2 &= y_5 \\
        \dot{y}_3 &= A a S(y_1 - y_2) - 2 a y_3 - a^2 y_0 \\
        \dot{y}_4 &= A a (\mu + a_2 J S(a_1 J y_0) + c) - 2 a y_4 - a^2 y_1 \\
        \dot{y}_5 &= B b (a_4 J S(a_3 J y_0)) - 2 b y_5 - b^2 y_2

    where :math:`S(v) = \frac{2 \nu_{max}}{1 + e^{r(v_0 - v)}}`.
    """
    y0, y1, y2, y3, y4, y5 = ys

    sigm_y1_y2 = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (y1 - y2))))
    sigm_y0_1  = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (p.a_1 * p.J * y0))))
    sigm_y0_3  = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (p.a_3 * p.J * y0))))

    return np.array([y3,
        y4,
        y5,
        p.A * p.a * sigm_y1_y2 - 2.0 * p.a * y3 - p.a ** 2 * y0,
        p.A * p.a * (p.mu + p.a_2 * p.J * sigm_y0_1 + c)
            - 2.0 * p.a * y4 - p.a ** 2 * y1,
        p.B * p.b * (p.a_4 * p.J * sigm_y0_3) - 2.0 * p.b * y5 - p.b ** 2 * y2,
                     ])


BVEPTheta = collections.namedtuple(
    typename='BVEPTheta',
    field_names='tau0 I1 x0'
)

bvep_default_theta = BVEPTheta(
    tau0=10.0, I1=3.1, x0=-3.5
)

def bvep_dfun(ys, c, p: BVEPTheta):
    x, z = ys
    x2 = x*x
    dx = 1 - x*x2 - 2*x2 - z + p.I1
    dz = (1/p.tau0)*(4*(x - p.x0) - z - c)
    return np.array([dx, dz])


# Montbrio-Pazo-Roxin
MPRTheta = collections.namedtuple(
    typename='MPRTheta',
    field_names='tau I Delta J eta cr cv'.split(' '))

mpr_default_theta = MPRTheta(
    tau=1.0,
    I=0.0,
    Delta=1.0,
    J=15.0,
    eta=-5.0,
    cr=1.0,
    cv=0.0
)

MPRState = collections.namedtuple(
    typename='MPRState',
    field_names='r V'.split(' '))

mpr_default_state = MPRState(r=0.0, V=-2.0)

def mpr_dfun(ys, c, p):
    r"""
    Montbrio-Pazo-Roxin mass model.

    Parameters
    ----------
    ys : array
        State vector of shape (2, ...).
        The components are:
        - r: Firing rate.
        - V: Membrane potential.
    c : tuple or array
        Coupling input. If tuple, contains (c_rate, c_volt).
    p : MPRTheta
        Parameters.

    Returns
    -------
    dys : array
        Derivatives of the state vector.

    Notes
    -----
    The equations are:

    .. math::
        \dot{r} &= \frac{1}{\tau} (\frac{\Delta}{\pi \tau} + 2 r V) \\
        \dot{V} &= \frac{1}{\tau} (V^2 + \eta + J \tau r + I + I_{c} - \pi^2 r^2 \tau^2)

    where :math:`I_c = c_r c_{rate} + c_v c_{volt}`.
    """
    r, V = ys

    # bound rate to be positive
    r = r * (r > 0)

    I_c = p.cr * c[0] + p.cv * c[1]

    return np.array([
        (1 / p.tau) * (p.Delta / (np.pi * p.tau) + 2 * r * V),
        (1 / p.tau) * (V ** 2 + p.eta + p.J * p.tau *
         r + p.I + I_c - (np.pi ** 2) * (r ** 2) * (p.tau ** 2))
    ])

def mpr_r_positive(rv, _):
    r, v = rv
    return np.array([ r*(r>0), v ])


BOLDTheta = collections.namedtuple(
    typename='BOLDTheta',
    field_names='tau_s,tau_f,tau_o,alpha,te,v0,e0,epsilon,nu_0,'
                'r_0,recip_tau_s,recip_tau_f,recip_tau_o,recip_alpha,'
                'recip_e0,k1,k2,k3'
)

def compute_bold_theta(
        tau_s=0.65,
        tau_f=0.41,
        tau_o=0.98,
        alpha=0.32,
        te=0.04,
        v0=4.0,
        e0=0.4,
        epsilon=0.5,
        nu_0=40.3,
        r_0=25.0,
    ):
    recip_tau_s = 1.0 / tau_s
    recip_tau_f = 1.0 / tau_f
    recip_tau_o = 1.0 / tau_o
    recip_alpha = 1.0 / alpha
    recip_e0 = 1.0 / e0
    k1 = 4.3 * nu_0 * e0 * te
    k2 = epsilon * r_0 * e0 * te
    k3 = 1.0 - epsilon
    return BOLDTheta(**locals())

bold_default_theta = compute_bold_theta()

def bold_dfun(sfvq, x, p: BOLDTheta):
    r"""
    Balloon-Windkessel BOLD model.

    Parameters
    ----------
    sfvq : array
        State vector of shape (4, ...).
        The components are:
        - s: Vasodilatory signal.
        - f: Blood inflow.
        - v: Blood volume.
        - q: Deoxyhemoglobin content.
    x : array
        Neural activity input.
    p : BOLDTheta
        Parameters.

    Returns
    -------
    dsfvq : array
        Derivatives of the state vector.

    Notes
    -----
    The equations are:

    .. math::
        \dot{s} &= x - \frac{s}{\tau_s} - \frac{f - 1}{\tau_f} \\
        \dot{f} &= s \\
        \dot{v} &= \frac{f - v^{1/\alpha}}{\tau_o} \\
        \dot{q} &= \frac{f \frac{1 - (1 - E_0)^{1/f}}{E_0} - v^{1/\alpha} \frac{q}{v}}{\tau_o}
    """
    s, f, v, q = sfvq
    ds = x - p.recip_tau_s * s - p.recip_tau_f * (f - 1)
    df = s
    dv = p.recip_tau_o * (f - v ** p.recip_alpha)
    dq = p.recip_tau_o * (f * (1 - (1 - p.e0) ** (1 / f)) * p.recip_e0
                          - v ** p.recip_alpha * (q / v))
    return np.array([ds, df, dv, dq])


DCMTheta = collections.namedtuple(typename='DCMTheta',
                                  field_names='A,B,C')

def dcm_dfun(x, u, p: DCMTheta):
    r"""Implements the classical bilinear DCM
    \dot{x} = (A + sum_j u_j B_j ) x + C u
    """
    return (p.A + np.sum(u * p.B,axis=-1)) @ x + p.C @ u


# TODO other models
# TODO codim3 https://gist.github.com/maedoc/01cea5cad9c833c56349392ee7d9b627


DopaTheta = collections.namedtuple(
    typename='dopaTheta',
    field_names='a, b, c, ga, gg, Eta, Delta, Iext, Ea, Eg, Sja, Sjg, tauSa, tauSg, alpha, beta, ud, k, Vmax, Km, Bd, Ad, tau_Dp, wi, we, wd, sigma')


DopaState = collections.namedtuple(
    typename='DopaState',
    field_names='r V u Sa Sg Dp')

dopa_default_theta = DopaTheta(
    a=0.04, b=5., c=140., ga=12., gg=12.,
    Delta=1., Eta=18., Iext=0., Ea=0., Eg=-80., tauSa=5., tauSg=5., Sja=0.8, Sjg=1.2,
    ud=12., alpha=0.013, beta=.4, k=10e4, Vmax=1300., Km=150., Bd=0.2, Ad=1., tau_Dp=500.,
    wi=1.e-4, we=1.e-4, wd=1.e-4, sigma=1e-3,
    )

dopa_default_initial_state = DopaState(
    r=0.03, V=-67.0, u=0.0, Sa=0.0, Sg=0.0, Dp=0.5)

def dopa_dfun(y, cy, p: DopaTheta):
    "Adaptive QIF model with dopamine modulation."

    r, V, u, Sa, Sg, Dp = y
    c_inh, c_exc, c_dopa = cy
    a, b, c, ga, gg, Eta, Delta, Iext, Ea, Eg, Sja, Sjg, tauSa, tauSg, alpha, beta, ud, k, Vmax, Km, Bd, Ad, tau_Dp, *_ = p

    dr = 2. * a * r * V + b * r - ga * Sa * r - gg * Sg * r + (a * Delta) / np.pi
    dV = a * V**2 + b * V + c + Eta - (np.pi**2 * r**2) / a + (Ad * Dp + Bd) * ga * Sa * (Ea - V) + gg * Sg * (Eg - V) + Iext - u
    du = alpha * (beta * V - u) + ud * r
    dSa = -Sa / tauSa + Sja * c_exc
    dSg = -Sg / tauSg + Sjg * c_inh
    dDp = (k * c_dopa - Vmax * Dp / (Km + Dp)) / tau_Dp
    
    return np.array([dr, dV, du, dSa, dSg, dDp])

def dopa_net_dfun(y, p):
    "Canonical form for network of dopa nodes."
    Ci, Ce, Cd, node_params = p
    r = y[0]
    c_inh = node_params.wi * Ci @ r
    c_exc = node_params.we * Ce @ r
    c_dopa = node_params.wd * Cd @ r
    return dopa_dfun(y, (c_inh, c_exc, c_dopa), node_params)

def dopa_r_positive(y, _):
    # same as mpr but keep separate name for now
    y = y.at[0].set( np.where(y[0]<0, 0, y[0]) )
    return y

def dopa_gfun_mulr(y, p):
    "Provide a multiplicative r, additive V gfun."
    r, *_ = y
    g = np.r_[r, 1, 0, 0, 0, 0] * p.sigma
    if r.ndim == 2:
        g = g.reshape((-1, 1))
    return g

def dopa_gfun_add(y, p):
    "Provides an additive noise gfun."
    return p.sigma
