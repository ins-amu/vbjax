import matplotlib.pyplot as pl
import jax.random as jr
import vbjax as vb

def example1():
    def network(x, p):
        c = 0.03*x.sum(axis=1)
        return vb.mpr_dfun(x, c, p)
    _, loop = vb.make_sde(dt=0.01, dfun=network, gfun=0.1)
    zs = vb.randn(2000, 2, 32)
    xs = loop(zs[0], zs[1:], vb.mpr_default_theta)
    return xs

def plot_example1():
    xs = example1()
    pl.figure(figsize=(10,4))
    pl.plot(xs[:,1], 'k', alpha=0.2)
    pl.ylabel('V(t)'), pl.xlabel('t (ms)')
    pl.grid(1);
    pl.title('Network mean membrane potential.')
    pl.tight_layout()
    pl.savefig('example1.jpg')
    pl.show()

# HA! only one example for now.

if __name__ == '__main__':
    plot_example1()
