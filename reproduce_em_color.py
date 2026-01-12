
import numpy as np
import jax
import jax.numpy as jnp
import vbjax

def em_color(f, g, dt, lam, x):
    """Euler-Maruyama for colored noise.
    Reference implementation from user.
    """
    i = 0
    nd = x.shape
    # Initialize e
    # g(i, x) is expected to return scalar or array compatible with x
    e = np.sqrt(g(i, x) * lam) * np.random.randn(*nd)
    E = np.exp(-lam * dt)
    while True:
        yield x, e
        i += 1
        x += dt * (f(i, x) + e)
        # Update e
        h = np.sqrt(g(i, x) * lam * (1 - E ** 2)) * np.random.randn(*nd)
        e = e * E + h

def test_em_color_repro():
    dt = 0.01
    lam = 1.0 # rate

    # Simple linear system: dx = -x dt + e dt
    # de = -lam e dt + sigma dW
    # Stationary variance of e is Q.
    # User code: e_init ~ N(0, g*lam).
    # So g*lam = Q.
    # h ~ N(0, g*lam*(1-E^2)).

    Q = 1.0
    g_val = Q / lam

    def f(i, x):
        return -x

    def g(i, x):
        return g_val

    x0 = np.array([1.0])
    gen = em_color(f, g, dt, lam, x0)

    xs = []
    es = []
    for _ in range(1000):
        x, e = next(gen)
        xs.append(x)
        es.append(e)

    xs = np.array(xs)
    es = np.array(es)

    print("Mean x:", np.mean(xs))
    print("Std x:", np.std(xs))
    print("Mean e:", np.mean(es))
    print("Std e:", np.std(es))
    print("Expected std e (approx):", np.sqrt(Q))

    # Check correlation of e
    # corr(e(t), e(t+dt)) = E = exp(-lam*dt)
    E = np.exp(-lam * dt)
    corr = np.corrcoef(es[:-1].flatten(), es[1:].flatten())[0,1]
    print(f"Correlation lag 1: {corr:.4f}, Expected: {E:.4f}")

if __name__ == "__main__":
    test_em_color_repro()
