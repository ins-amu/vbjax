import numpy as np
import jax
import jax.test_util
import jax.dlpack
import jax.numpy as jnp
import scipy.sparse
import vbjax


def _test_spmv(spmv, A, n):
    nx = np.random.randn(n)
    jx = jnp.array(nx)

    # test the function itself
    jb = spmv(jx)
    nb = A @ nx
    np.testing.assert_allclose(jb, nb, 1e-6, 1e-6)

    # now its gradient
    jax.test_util.check_grads(spmv, (jx,), order=1, modes=('rev',))


def test_csr_scipy():
    n = 10
    A = scipy.sparse.random(n, n, density=0.1).tocsr()
    spmv = vbjax.make_spmv(A)
    _test_spmv(spmv, A, n)


def test_csr_scipy_symm():
    n = 10
    A = scipy.sparse.random(n, n, density=0.1).tocsr()
    A += A.T
    spmv = vbjax.make_spmv(A, is_symmetric=True)
    _test_spmv(spmv, A, n)


if __name__ == '__main__':
    test_csr_scipy()
    test_csr_scipy_symm()
