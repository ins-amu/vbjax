import jax
import jax.numpy as np

def jacobi(A, b, w=2.0/3.0, tol=1e-9, max_iters=100):
    """
    Jacobi iterative linear solver for Ax = b.

    Parameters
    ----------
    A : array
        Matrix A.
    b : array
        Vector b.
    w : float
        Relaxation parameter.
    tol : float
        Tolerance.
    max_iters : int
        Maximum number of iterations.

    Returns
    -------
    x : array
        Solution vector.
    n_iter : int
        Number of iterations performed.
    """

    # Precompute diagonal inverse and LU
    diag_A = np.diag(A)
    inv_diag_A = 1.0 / diag_A
    w_m_invD = w * np.diag(inv_diag_A)
    LU = A - np.diag(diag_A)

    x0 = b

    def cond(state):
        x, dx, n_iter = state
        return (n_iter < max_iters) & (np.linalg.norm(dx) > tol)

    def body(state):
        x, dx, n_iter = state
        xn = w_m_invD @ (b - LU @ x) + (1.0 - w) * x
        dx = x - xn
        return xn, dx, n_iter + 1

    dx = np.ones_like(x0)
    # Using jax.lax.while_loop for the iterative solver
    x, dx, n_iter = jax.lax.while_loop(cond, body, (x0, dx, 0))

    return x, n_iter


def _theta_dy(f, j, h, th, y0, y1, f_y0, pars, tol=1e-4):
    J = np.eye(y1.size) - h * th * j(y1, pars)
    F = y0 + h * (th * f(y1, pars) + f_y0) - y1

    dy, _ = jacobi(J, F, w=2.0/3.0, tol=tol, max_iters=10)
    return dy

def _compute_noise(gfun, x, p, sqrt_dt, z_t):
    g = gfun(x, p)
    return g * sqrt_dt * z_t


def make_implicit_sde(dt, dfun, jfun, gfun, th=0.5, tol=1e-4, max_iters=10):
    """
    Construct an implicit SDE integrator using the Theta method.

    Parameters
    ----------
    dt : float
        Time step.
    dfun : callable
        Drift function dfun(x, p).
    jfun : callable
        Jacobian of drift function jfun(x, p).
    gfun : callable or float
        Diffusion function gfun(x, p) or constant sigma.
    th : float
        Theta parameter (0.5 for Crank-Nicolson, 1.0 for backward Euler).
    tol : float
        Tolerance for the Newton-Jacobi iteration.
    max_iters : int
        Maximum iterations for the Newton loop.

    Returns
    -------
    step : callable
        Step function step(x, z_t, p).
    loop : callable
        Loop function loop(x0, zs, p).
    """

    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    sqrt_dt = np.sqrt(dt)

    def step(x, z_t, p):
        # Euler guess
        f_val = dfun(x, p)
        f_y0 = (1 - th) * f_val
        y1_euler = x + dt * f_val # Initial guess using explicit Euler

        # Refine if implicit
        def refine(y1):
             # First Newton step
             # We use y1 (Euler guess) to compute Jacobian and Residual
             dy = _theta_dy(dfun, jfun, dt, th, x, y1, f_y0, p, tol=tol)
             y1 = y1 + dy

             def refinement_cond(state):
                 y1, dy, n_iter = state
                 return (n_iter < max_iters) & (np.linalg.norm(dy) > tol)

             def refinement_body(state):
                 y1, _, n_iter = state
                 dy = _theta_dy(dfun, jfun, dt, th, x, y1, f_y0, p, tol=tol)
                 y1 = y1 + dy
                 return y1, dy, n_iter + 1

             init_state = (y1, dy, 1)
             final_state = jax.lax.while_loop(refinement_cond, refinement_body, init_state)
             return final_state[0]

        y1 = jax.lax.cond(th > 0.0, refine, lambda x: x, y1_euler)

        noise = _compute_noise(gfun, x, p, sqrt_dt, z_t)
        y1 = y1 + noise

        return y1

    @jax.jit
    def loop(x0, zs, p):
        def op(x, z):
            x = step(x, z, p)
            return x, x
        _, xs = jax.lax.scan(op, x0, zs)
        return xs

    return step, loop
