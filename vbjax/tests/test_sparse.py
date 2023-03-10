import numpy as np
import jax
import jax.test_util
import jax.dlpack
import jax.numpy as jnp
import scipy.sparse


def test_csr_scipy():
    n = 10
    A: scipy.sparse.csr_matrix = scipy.sparse.random(n, n, density=0.1).tocsr()
    nx = np.random.randn(n)
    jx = jnp.array(nx)

    @jax.custom_vjp
    def matvec(x):
        nb = A @ np.from_dlpack(x)
        db = jax.dlpack.from_dlpack(nb.__dlpack__())
        return db

    def matvec_tr(x):
        nb = A.T @ np.from_dlpack(x)
        db = jax.dlpack.from_dlpack(nb.__dlpack__())
        return db

    def matvec_fwd(x):
        return matvec(x), None

    def matvec_bwd(res, g):
        return matvec_tr(g),

    matvec.defvjp(matvec_fwd, matvec_bwd)

    # test the function itself
    jb = matvec(jx)
    nb = A @ nx
    np.testing.assert_allclose(jb, nb, 1e-6, 1e-6)

    # now its gradient
    jax.test_util.check_grads(matvec, (jx,), order=1, modes=('rev',))

if __name__ == '__main__':
    test_csr_scipy()
