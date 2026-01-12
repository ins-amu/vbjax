
import jax
import jax.numpy as np
import vbjax
import pytest

def test_jacobi():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x, n_iter = vbjax.jacobi(A, b, tol=1e-6)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, atol=1e-5)

def test_make_implicit_sde_linear():
    # dy = -k * y * dt
    k = 1.0
    def f(y, k):
        return -k * y

    def j_fn(y, k):
        return -k * np.eye(y.size)

    y0 = np.array([1.0])
    dt = 0.1
    tf = 1.0

    # 0.5 = Crank-Nicolson -> trapezoidal rule
    step, loop = vbjax.make_implicit_sde(dt, f, j_fn, 0.0, th=0.5)

    # Create noise (zeros since deterministic test)
    n_steps = int(tf / dt)
    zs = np.zeros((n_steps, 1))

    ys = loop(y0, zs, k)

    ts = np.arange(1, n_steps + 1) * dt
    exact = y0 * np.exp(-ts) # Actually Trapezoidal rule approximation will differ slightly from exact exp
    # Trapezoidal: y_{n+1} = y_n + h/2 (f_n + f_{n+1}) = y_n - k*h/2 (y_n + y_{n+1})
    # y_{n+1} (1 + kh/2) = y_n (1 - kh/2)
    # y_{n+1} = y_n * (1 - kh/2) / (1 + kh/2)

    factor = (1 - k*dt/2) / (1 + k*dt/2)
    expected_numeric = y0 * (factor ** np.arange(1, n_steps + 1))

    error = np.abs(ys.flatten() - expected_numeric)
    assert np.max(error) < 1e-5

    # Also check against exact just to be sure it's close
    error_exact = np.abs(ys.flatten() - exact)
    assert np.max(error_exact) < 1e-2


def test_make_implicit_sde_autodiff():
    # dy = -y^3
    def f(y, p):
        return -y**3

    # Auto-diff Jacobian
    j_fn = jax.jacfwd(f)

    y0 = np.array([1.0])
    dt = 0.1
    tf = 1.0

    step, loop = vbjax.make_implicit_sde(dt, f, j_fn, 0.0, th=1.0) # Backward Euler

    n_steps = int(tf / dt)
    zs = np.zeros((n_steps, 1))

    ys = loop(y0, zs, None)

    # Backward Euler: y_{n+1} = y_n - h * y_{n+1}^3
    # Solve y_{n+1} + h * y_{n+1}^3 - y_n = 0
    # Check consistency
    for i in range(n_steps):
        prev = y0 if i == 0 else ys[i-1]
        curr = ys[i]
        res = curr + dt * curr**3 - prev
        assert np.abs(res) < 1e-4

if __name__ == "__main__":
    test_jacobi()
    test_make_implicit_sde_linear()
    test_make_implicit_sde_autodiff()
