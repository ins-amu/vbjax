import jax
import jax.numpy as np
import jax.random as jr
import vbjax as vb
import pytest

def test_sde_colored_noise():
    dt = 0.01
    lam = 1.0 # noise rate
    sigma = 1.0

    # Define linear system dx = -x dt + e dt
    # where e is colored noise with correlation time 1/lam
    # Stationary variance of e is sigma^2 * lam.

    # Use make_sde with noise_rate
    step, loop = vb.make_sde(dt, lambda x, p: -x, sigma, noise_rate=lam)

    key = jr.PRNGKey(42)
    n_steps = 100000
    zs = jr.normal(key, (n_steps,))

    x0 = np.array(0.0)
    # Initialize e from stationary dist: N(0, sigma^2 * lam)
    # sigma=1, lam=1 -> std=1
    e0 = jr.normal(jr.PRNGKey(0))

    # State is (x, e)
    xe0 = (x0, e0)

    # Run loop
    xes = loop(xe0, zs, None)

    # unpack
    # xes should be ((xs...), (es...)) if return_euler=False
    xs, es = xes

    print(f"Mean x: {np.mean(xs)}")
    print(f"Std x: {np.std(xs)}")
    print(f"Mean e: {np.mean(es)}")
    print(f"Std e: {np.std(es)}")

    # Check correlation of e
    E = np.exp(-lam * dt)
    corr = np.corrcoef(es[:-1], es[1:])[0,1]
    print(f"Correlation e lag 1: {corr:.4f}, Expected: {E:.4f}")

    # With 100,000 steps, effective sample size is approx 500. 1/sqrt(500) ~ 0.045.
    # We use 0.15 as tolerance (3 sigma)
    assert np.abs(np.mean(es)) < 0.15
    assert np.abs(np.std(es) - 1.0) < 0.1
    assert np.abs(corr - E) < 0.01

def test_sdde_colored_noise():
    dt = 0.01
    lam = 1.0
    sigma = 1.0
    nh = 10

    # Define SDDE dx = -x(t) dt + e dt

    step, loop = vb.make_sdde(dt, nh, lambda buf, x, t, p: -x, sigma, noise_rate=lam)

    key = jr.PRNGKey(42)
    n_steps = 100000

    total_len = nh + 1 + n_steps
    buf = np.zeros((total_len,))

    zs = jr.normal(key, (total_len,))
    buf = buf.at[nh+1:].set(zs[nh+1:])

    e0 = jr.normal(jr.PRNGKey(0))

    # Initial state (buf, e)
    buf_e = (buf, e0)

    (buf_res, ne), nxs = loop(buf_e, None)

    xs = nxs
    print(f"Mean x (sdde): {np.mean(xs)}")
    print(f"Std x (sdde): {np.std(xs)}")

    # x should behave similarly to SDE case.
    # Std x for SDE is approx 0.67

    assert np.abs(np.std(xs) - 0.67) < 0.1

if __name__ == "__main__":
    test_sde_colored_noise()
    test_sdde_colored_noise()
