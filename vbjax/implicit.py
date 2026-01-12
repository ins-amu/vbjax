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
    J = np.eye(y1.size) - h * th * j(y1, *pars)
    F = y0 + h * (th * f(y1, *pars) + f_y0) - y1

    dy, _ = jacobi(J, F, w=2.0/3.0, tol=tol, max_iters=10)
    return dy


def theta(f, j, y0, h, tf, *pars, th=0.5, sigma=0.0, tol=1e-4, key=None):
    """
    Implicit Theta integration scheme.

    Parameters
    ----------
    f : callable
        Drift function f(y, *pars).
    j : callable
        Jacobian function j(y, *pars).
    y0 : array
        Initial state.
    h : float
        Time step.
    tf : float
        Final time.
    *pars : list
        Parameters passed to f and j.
    th : float
        Theta parameter (0.5 for trapezoidal/Crank-Nicolson, 1.0 for backward Euler).
    sigma : float or array
        Noise standard deviation.
    tol : float
        Tolerance for the Newton-Jacobi iteration.
    key : jax.random.PRNGKey, optional
        Random key for noise generation. If None, a default key is used.

    Returns
    -------
    ts : array
        Time points.
    ys : array
        Trajectory.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    num_steps = int(tf / h)
    add_noise = np.any(sigma > 0.0)

    def scan_body(carry, _):
        y0, rng_key = carry

        f_y0 = (1 - th) * f(y0, *pars)
        y1 = y0 + h * f_y0

        def refine(y1):
             dy = _theta_dy(f, j, h, th, y0, y0, f_y0, pars, tol=tol)
             y1 = y0 + dy

             def refinement_cond(state):
                 y1, dy, n_iter = state
                 return (n_iter < 10) & (np.linalg.norm(dy) > tol)

             def refinement_body(state):
                 y1, _, n_iter = state
                 dy = _theta_dy(f, j, h, th, y0, y1, f_y0, pars, tol=tol)
                 y1 = y1 + dy
                 return y1, dy, n_iter + 1

             init_state = (y1, dy, 1)
             final_state = jax.lax.while_loop(refinement_cond, refinement_body, init_state)
             return final_state[0]

        y1 = jax.lax.cond(th > 0.0, refine, lambda x: x, y1)

        rng_key, step_key = jax.random.split(rng_key)
        noise = jax.lax.cond(
            add_noise,
            lambda k: np.sqrt(sigma) * jax.random.normal(k, y0.shape),
            lambda k: np.zeros_like(y0),
            step_key
        )
        y1 = y1 + noise

        return (y1, rng_key), y1

    _, ys = jax.lax.scan(scan_body, (y0, key), None, length=num_steps)

    # Prepend y0
    ys = np.concatenate([y0[None, ...], ys], axis=0)
    ts = np.arange(num_steps + 1) * h

    return ts, ys
