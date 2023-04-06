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
    field_names='tau0 I1 eta'
)

bvep_default_theta = BVEPTheta(
    tau0=10.0, I1=3.1, eta=-3.5
)

def bvep_dfun(ys, c, p: BVEPTheta):
    z, x = ys
    x2 = x*x
    dx = 1 - x*x2 - 2*x2 - z + p.I1
    dz = (1/p.tau0)*(4*(x - p.eta) - z - c)
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
    r, V = ys

    I_c = p.cr * c[0] + p.cv * c[1]

    return np.array([
        (1 / p.tau) * (p.Delta / (np.pi * p.tau) + 2 * r * V),
        (1 / p.tau) * (V ** 2 + p.eta + p.J * p.tau *
         r + p.I + I_c - (np.pi ** 2) * (r ** 2) * (p.tau ** 2))
    ])
