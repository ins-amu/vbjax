import pylab as pl
import vbjax as vb
import jax.numpy as jp

n = 5

# assume damped dynamics at each node
A = -jp.identity(n)

# two conditions
B = jp.zeros((n, n, 2))
C = jp.zeros((n, B.shape[2]))

# first condition induce oscillation between nodes 1 & 2
B = B.at[0,1,0].set(1)
B = B.at[1,0,0].set(-1)

# second condition induce faster oscillation between nodes 3 & 4
B = B.at[3,2,1].set(3)
B = B.at[2,3,1].set(-3)

# second condition shifts node 5 fixed point up by 0.5
C = C.at[4, 1].set(0.5)


def dcm(x, u):
    p = vb.DCMTheta(A=A, B=B, C=C)
    return vb.dcm_dfun(x, u, p)

_, loop = vb.make_ode(0.2, dcm)
ts = jp.r_[:64]
x0 = jp.ones(5)

conditions = [
    jp.r_[0, 0], # rest
    jp.r_[1, 0], # condition 1
    jp.r_[0, 1], # condition 2
    ]
titles = 'rest,osc 1 2,fast osc 3 4 shift 5'.split(',')
pl.figure(figsize=(8, 8))
for i, cond in enumerate(conditions):
    pl.subplot(1, 3, i + 1)
    xs = loop(x0, ts, cond)
    pl.plot(xs + jp.r_[:n], 'k')
    pl.grid(1)
    pl.title(titles[i])
    pl.xlabel('time (au)')
pl.tight_layout()

pl.show()
