import numpy as np
import scipy.sparse
import jax
import jax.experimental.sparse as jsp
import vbjax as vb


def make_spmv(A, is_symmetric=False, use_scipy=False):
    """
    Make a closure for a general sparse matrix-vector multiplication.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Constant sparse matrix.
    is_symmetric : bool, optional, default False
        Whether matrix is symmetric.
    use_scipy: bool, optional, default False
        Use scipy.

    Returns
    -------
    spmv : function
        Function implementing spase matrix vector multiply with
        support for gradients in Jax.

    """
    AT = A.T.copy()
    @jax.custom_vjp
    def matvec(x):          return vb.to_jax(A @ vb.to_np(x))
    def matvec_tr(x):       return vb.to_jax(AT @ vb.to_np(x))
    def matvec_fwd(x):      return matvec(x), None
    def matvec_bwd(res, g): return matvec(g) if is_symmetric else matvec_tr(g),
    matvec.defvjp(matvec_fwd, matvec_bwd)
    return matvec


def csr_to_jax_bcoo(A: scipy.sparse.csr_matrix) :
    "Convert CSR format to batched COO format."
    # first convert CSR to COO
    coo = A.tocoo()
    # now convert to batched COO
    data = jax.numpy.array(coo.data)
    indices = jax.numpy.array(np.c_[coo.row, coo.col])
    shape = A.shape
    return jsp.BCOO((data, indices), shape=shape)


def make_sg_spmv(A: scipy.sparse.csr_matrix):
    "Make a SpMV kernel w/ generic scatter-gather operations."
    import jax.numpy as np
    nrow = A.shape[0]
    col = np.array(A.indices)
    row = np.zeros_like(col)
    for i, (r0, r1) in enumerate(zip(A.indptr[:-1], A.indptr[1:])):
        row = row.at[r0:r1].set(i)
    dat = np.array(A.data)
    def spmv(x):
        return np.zeros(nrow).at[row].add(dat * x[col])
    return spmv