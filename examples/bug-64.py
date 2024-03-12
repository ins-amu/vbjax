import collections
import networkx as nx
import matplotlib.pyplot as plt
import jax.numpy as jnp
import vbjax as vb

KMTheta = collections.namedtuple(typename="KMTheta", field_names="G omega".split(" "))
km_default_theta = KMTheta(G=0.05, omega=1.0)
KMState = collections.namedtuple(typename="KMState", field_names="x".split(" "))

def km_dfun(x, c, p: KMTheta):
    "Kuramoto model"
    dx = p.omega + jnp.vdot(p.G, c) # or just  p.G * c
    return dx

def network(x, p):
    weights, node_params = p
    c = jnp.sum(weights * jnp.sin(x - x[:, None]), axis=1)
    dx = km_dfun(x, c, node_params)
    return dx


def get_ts(params, dt=0.1, T=50.0, G=0.0, sigma=0.1):
    '''Run the Kuramoto model'''
    omega, weights, par = params    
    nn = weights.shape[0]
    G = jnp.ones(nn) * G
    _, loop = vb.make_sde(dt, dfun=network, gfun=sigma)
    par = par._replace(G=G, omega=omega)
    nt = int(T / dt)
    zs = vb.randn(nt, nn) * 2 * jnp.pi
    xs = loop(zs[0], zs[1:], (weights, par))
    ts = jnp.linspace(0, nt * dt, len(xs))
    return xs, ts 
    
    
nn = 3
weights = nx.to_numpy_array(nx.complete_graph(nn))
dt = 0.1

omega = jnp.abs(vb.randn(nn) * 1.0)
print('omega values are', omega)

plt.figure()
for i, sigma in enumerate([0.0, 0.01, 0.1]):
    xs, ts = get_ts((omega, weights, km_default_theta), dt=dt, G=0.1, sigma=sigma)
    plt.subplot(3, 1, i + 1)
    plt.plot(ts, jnp.sin(xs))
    print(i, 'sigma=', sigma, jnp.sum(jnp.diff(xs,axis=0)) )
plt.show()
