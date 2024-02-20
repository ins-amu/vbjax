
import time
import numpy
import jax
import jax.test_util
import jax.dlpack
import jax.numpy as np
import scipy.sparse
import vbjax as vb
import pytest


def _test_spmv(spmv, A, n):
    nx = numpy.random.randn(n)
    jx = np.array(nx)

    # test the function itself
    jb = spmv(jx)
    nb = A @ nx
    numpy.testing.assert_allclose(jb, nb, 1e-4, 1e-6)

    # now its gradient
    jax.test_util.check_grads(spmv, (jx,), order=1, modes=('rev',), atol=0.02, rtol=0.002)


def test_csr_scipy():
    n = 10
    A = scipy.sparse.random(n, n, density=0.1).tocsr()
    spmv = vb.make_spmv(A)
    _test_spmv(spmv, A, n)


def test_csr_scipy_symm():
    n = 10
    A = scipy.sparse.random(n, n, density=0.1).tocsr()
    A += A.T
    spmv = vb.make_spmv(A, is_symmetric=True)
    _test_spmv(spmv, A, n)


def test_sg_spmv():
    n = 100
    A = scipy.sparse.random(n, n, density=0.1).tocsr()
    spmv = vb.make_sg_spmv(A, False)
    _test_spmv(spmv, A, n)


@pytest.mark.skipif(jax.local_devices()[0].platform == 'gpu', reason='skip gpu')
def test_sg_spmv_pmap():
    n = 100
    A = scipy.sparse.random(n, n, density=0.1).tocsr()
    nrow = A.shape[0]
    col = np.array(A.indices)
    row = np.array(numpy.concatenate([
        i * numpy.ones(n, 'i')
        for i, n in enumerate(numpy.diff(A.indptr))]))
    dat = np.array(A.data)
    spmv = vb.make_sg_spmv(A, True)
    _test_spmv(spmv, A, n)


def bench_csr_to_bcoo():
    n = 1000
    A = scipy.sparse.random(n, n, density=0.1).tocsr()
    spmv = vb.make_spmv(A)
    jx = jax.numpy.r_[:n].astype('f')
    jb1 = spmv(jx)

    # now convert to bcoo
    jA = vb.csr_to_jax_bcoo(A)
    jspmv = jax.jit(lambda x: jA @ x)
    jb2 = jspmv(jx)

    np.testing.assert_allclose(jb1, jb2, 1e-6, 1e-6)

    for f, name in zip((spmv, jspmv), 'csr bcoo'.split(' ')):
        t0 = time.time()
        for _ in range(100):
            jb2 = f(jx)
        t1 = time.time()
        print(f'{name}: {t1 - t0:.3f} s')


def create_sample_data(n=1000, density_pct=10):
    A = scipy.sparse.random(n, n, density=density_pct/100).tocsr()
    jx = jax.numpy.r_[:n].astype('f')
    return A, jx


# some performance testing values
_perf_args = 'n,density_pct,grad,impl,jit'
_perf_values = [(1000, 10), (10_000, 0.02)]

# we want to test each of the above values with grad on and off
_perf_values = [vals + (flag, impl, jit)
                for flag in (True, False)
                for impl in 'scipy jaxbcoo'.split(' ')
                for jit in (True, False)
                for vals in _perf_values]

@pytest.mark.parametrize(_perf_args, _perf_values)
def test_perf_jbcoo(benchmark, n, density_pct, grad, impl, jit):
    A, x = create_sample_data(n=n, density_pct=density_pct)

    if impl == 'scipy': # TODO enum
        spmv1 = vb.make_spmv(A)
    elif impl == 'jaxbcoo':
        jA = vb.csr_to_jax_bcoo(A)
        spmv1 = lambda x: jA @ x
    else:
        raise ValueError(impl)
    assert callable(spmv1)
    
    if grad:
        spmv2 = jax.grad(lambda x: np.sum(spmv1(x)))
    else:
        spmv2 = spmv1
    assert callable(spmv2)

    if jit and impl not in ('scipy', ):
        spmv3 = jax.jit(spmv2)
        spmv3(x)
    else:
        spmv3 = spmv2
    assert callable(spmv3)

    benchmark(lambda : spmv3(x))    
