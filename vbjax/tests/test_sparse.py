import time
import numpy as np
import jax
import jax.test_util
import jax.dlpack
import jax.numpy as jnp
import jax.experimental.sparse as jsp
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


def _csr_to_jax_bcoo(A: scipy.sparse.csr_matrix):
    "Convert CSR format to batched COO format."
    # first convert CSR to COO
    coo = A.tocoo()
    # now convert to batched COO
    data = jax.numpy.array(coo.data)
    indices = jax.numpy.array(np.c_[coo.row, coo.col])
    shape = A.shape
    return jsp.BCOO((data, indices), shape=shape)


def bench_csr_to_bcoo():
    n = 1000
    A = scipy.sparse.random(n, n, density=0.1).tocsr()
    spmv = vbjax.make_spmv(A)
    jx = jax.numpy.r_[:n].astype('f')
    jb1 = spmv(jx)

    # now convert to bcoo
    jA = _csr_to_jax_bcoo(A)
    jspmv = jax.jit(lambda x: jA @ x)
    jb2 = jspmv(jx)

    np.testing.assert_allclose(jb1, jb2, 1e-6, 1e-6)

    for f, name in zip((spmv, jspmv), 'csr bcoo'.split(' ')):
        t0 = time.time()
        for _ in range(100):
            jb2 = f(jx)
        t1 = time.time()
        print(f'{name}: {t1 - t0:.3f} s')
    

if __name__ == '__main__':
    bench_csr_to_bcoo()
