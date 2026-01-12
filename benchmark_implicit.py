
import jax
import jax.numpy as np
import vbjax
import time

def benchmark():
    # Stiff system example: Robertson problem or simple stiff linear
    # Let's use a stiff linear system: dy/dt = -1000 y
    k = 1000.0
    dt = 0.001 # 1/k.  Explicit stability limit is 2/k = 0.002.
    # If we use dt = 0.01, explicit should explode, implicit should work.

    def f(y, k):
        return -k * y

    def j_fn(y, k):
        return -k * np.eye(y.size)

    def g(y, k):
        return 0.1

    y0 = np.ones(100) # 100 dimensions
    tf = 1.0

    # 1. Heun (Explicit)
    # We use a small dt for stability or just measure speed
    dt_explicit = 0.0001
    n_explicit = int(tf / dt_explicit)
    zs_explicit = jax.random.normal(jax.random.PRNGKey(0), (n_explicit, 100))

    _, loop_heun = vbjax.make_sde(dt_explicit, f, g)

    # Compile
    loop_heun(y0, zs_explicit, k).block_until_ready()

    t0 = time.time()
    for _ in range(10):
        loop_heun(y0, zs_explicit, k).block_until_ready()
    t_heun = (time.time() - t0) / 10
    print(f"Heun (dt={dt_explicit}, steps={n_explicit}): {t_heun*1000:.2f} ms")

    # 2. Implicit (Theta=0.5)
    # Use larger dt
    dt_implicit = 0.01
    n_implicit = int(tf / dt_implicit)
    zs_implicit = jax.random.normal(jax.random.PRNGKey(0), (n_implicit, 100))

    step_imp, loop_imp = vbjax.make_implicit_sde(dt_implicit, f, j_fn, g, th=0.5)

    # Compile
    loop_imp(y0, zs_implicit, k).block_until_ready()

    t0 = time.time()
    for _ in range(10):
        loop_imp(y0, zs_implicit, k).block_until_ready()
    t_imp = (time.time() - t0) / 10
    print(f"Implicit (dt={dt_implicit}, steps={n_implicit}): {t_imp*1000:.2f} ms")

    print(f"Speedup factor (Total Time): {t_heun / t_imp:.2f}x")

if __name__ == "__main__":
    benchmark()
