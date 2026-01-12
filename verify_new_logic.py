
import jax
import jax.numpy as np
import jax.random as jr
import vbjax

def make_sde_mock(dt, dfun, gfun, noise_rate=None):
    sqrt_dt = np.sqrt(dt)

    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    if noise_rate is None:
        # Standard SDE (simplified for mock)
        def step(x, z_t, p):
            noise = gfun(x, p) * sqrt_dt * z_t
            d1 = dfun(x, p)
            xi = x + dt * d1 + noise
            d2 = dfun(xi, p)
            nx = x + dt * 0.5 * (d1 + d2) + noise
            return nx

        def loop(x0, zs, p):
            def op(x, z):
                x = step(x, z, p)
                return x, x
            _, xs = jax.lax.scan(op, x0, zs)
            return xs
        return step, loop
    else:
        # Colored noise
        # E = exp(-lam * dt)
        E = np.exp(-noise_rate * dt)
        # 1 - E^2
        one_minus_E2 = 1.0 - E**2
        # We assume gfun returns sigma.
        # But for colored noise, we need to match the variance.
        # If gfun is sigma, then variance of 'white' noise is sigma^2.
        # If we want colored noise 'e' to have variance sigma^2?
        # Or do we want 'e' to scale such that it matches something else?
        # In em_color: e = sqrt(g * lam) * N(0,1). g is usually Q/lam.
        # So var(e) = g*lam = Q.
        # If gfun returns sigma, and we assume sigma^2 = Q = g*lam.
        # Then g = sigma^2 / lam.
        # But let's assume user provides gfun corresponding to sigma.
        # And we want e to have variance sigma^2?
        # Or maybe gfun IS sigma_e?

        # Let's assume gfun returns the parameter corresponding to 'g' in em_color IF the user
        # is porting code. But vbjax users expect gfun to be diffusion coef.
        # If I want to implement em_color exactly as requested:
        # "the old tvb-algo implements colored noise ... let's try adding support for that"
        # The user provided em_color code uses 'g' inside sqrt.
        # If I use gfun as sigma, then gfun^2 corresponds to g in em_color (maybe without lam factor).

        # Let's stick to my plan:
        # h = gfun * sqrt(lam * (1 - E^2)) * z_t
        # This implies var(h) = gfun^2 * lam * (1 - E^2).
        # Stationary var(e) = var(h) / (1 - E^2) = gfun^2 * lam.
        # So sigma_e = gfun * sqrt(lam).

        # If noise_rate (lam) is large, sigma_e is large.

        def step(xe, z_t, p):
            x, e = xe
            # x update: Heun with additive noise e * dt
            # noise term for x is e * dt

            # Use vbjax.loops.heun_step if available, but here I inline for mock
            noise_x = e * dt

            d1 = dfun(x, p)
            xi = x + dt * d1 + noise_x
            d2 = dfun(xi, p)
            nx = x + dt * 0.5 * (d1 + d2) + noise_x

            # e update
            # We use x (old x) to compute diffusion?
            sigma = gfun(x, p)
            h = sigma * np.sqrt(noise_rate * one_minus_E2) * z_t
            ne = e * E + h

            return (nx, ne)

        def loop(xe0, zs, p):
            def op(xe, z):
                xe = step(xe, z, p)
                return xe, xe
            _, xes = jax.lax.scan(op, xe0, zs)
            # Unpack?
            # xes is ((nx_0, nx_1...), (ne_0, ne_1...))
            return xes

        return step, loop

def test_mock():
    dt = 0.01
    lam = 1.0 # noise rate
    sigma = 1.0 # gfun return
    # Expected var(e) = sigma^2 * lam = 1.0 * 1.0 = 1.0.

    noise_rate = lam
    step, loop = make_sde_mock(dt, lambda x,p: -x, sigma, noise_rate=noise_rate)

    key = jr.PRNGKey(42)
    zs = jr.normal(key, (100000,))

    x0 = np.array(0.0)
    # Initialize e0 from stationary dist
    e0 = sigma * np.sqrt(lam) * jr.normal(jr.PRNGKey(0))

    xe0 = (x0, e0)

    xes = loop(xe0, zs, None)
    xs, es = xes

    print("Mean x:", np.mean(xs))
    print("Std x:", np.std(xs))
    print("Mean e:", np.mean(es))
    print("Std e:", np.std(es))

    # Check correlation of e
    E = np.exp(-lam * dt)
    corr = np.corrcoef(es[:-1], es[1:])[0,1]
    print(f"Correlation e lag 1: {corr:.4f}, Expected: {E:.4f}")

    # Expected Std e = 1.0
    print(f"Expected Std e: {sigma * np.sqrt(lam)}")

if __name__ == "__main__":
    test_mock()
