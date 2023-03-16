import numpy as np
import jax.dlpack
import scipy.sparse


_to_np = lambda x: np.from_dlpack(x)
_to_jax = lambda x: jax.dlpack.from_dlpack(x.__dlpack__())


def make_spmv(A, is_symmetric=False):
    """
    Make a closure for a general sparse matrix-vector multiplication.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Constant sparse matrix.
    is_symmetric : bool, optional, default False
        Whether matrix is symmetric.

    Returns
    -------
    spmv : function
        Function implementing spase matrix vector multiply with
        support for gradients in Jax.

    """
    AT = A.T.copy()
    @jax.custom_vjp
    def matvec(x):          return _to_jax(A @ _to_np(x))
    def matvec_tr(x):       return _to_jax(AT @ _to_np(x))
    def matvec_fwd(x):      return matvec(x), None
    def matvec_bwd(res, g): return matvec(g) if is_symmetric else matvec_tr(g),
    matvec.defvjp(matvec_fwd, matvec_bwd)
    return matvec
