
import jax
import jax.numpy as np
import vbjax
import pytest

def test_jacobi():
    # Solve Ax = b
    # A = [[4, 1], [1, 3]]
    # b = [1, 2]
    # Exact solution:
    # 4x + y = 1
    # x + 3y = 2
    # x = 2 - 3y
    # 4(2-3y) + y = 1 => 8 - 12y + y = 1 => -11y = -7 => y = 7/11
    # x = 2 - 21/11 = 1/11

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    x, n_iter = vbjax.jacobi(A, b, tol=1e-6)

    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, atol=1e-5)
    print(f"Jacobi solved in {n_iter} iterations. x={x}")


def test_theta_linear():
    # Test implicit scheme on a simple linear ODE: y' = -y
    # Analytical solution: y(t) = y0 * exp(-t)

    k = 1.0
    def f(y, k):
        return -k * y

    def j_fn(y, k):
        # Jacobian is scalar -k
        # If y is vector, Jacobian is diagonal -k * I
        return -k * np.eye(y.size)

    y0 = np.array([1.0])
    h = 0.1
    tf = 1.0

    # Run with Theta=0.5 (Crank-Nicolson)
    ts, ys = vbjax.theta(f, j_fn, y0, h, tf, k, th=0.5)

    exact = y0 * np.exp(-ts)
    error = np.abs(ys.flatten() - exact)

    print(f"Max error (Theta=0.5): {np.max(error)}")
    assert np.max(error) < 1e-2


def test_theta_auto_jacobian():
    # Same as above but using jax.jacfwd
    k = 1.0
    def f(y, k):
        return -k * y

    j_fn = jax.jacfwd(f, argnums=0)

    y0 = np.array([1.0])
    h = 0.1
    tf = 1.0

    ts, ys = vbjax.theta(f, j_fn, y0, h, tf, k, th=0.5)

    exact = y0 * np.exp(-ts)
    error = np.abs(ys.flatten() - exact)

    print(f"Max error (Theta=0.5, AutoDiff): {np.max(error)}")
    assert np.max(error) < 1e-2

if __name__ == "__main__":
    test_jacobi()
    test_theta_linear()
    test_theta_auto_jacobian()
