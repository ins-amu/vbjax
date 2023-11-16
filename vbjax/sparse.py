import numpy
import scipy.sparse
import jax
import jax.numpy as np
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

def _pad_up(x):
    npad = (x.size // vb.cores + 1) * vb.cores - x.size
    out = np.r_[x, np.zeros(npad,'i')]
    return out

def make_sg_spmv(A: scipy.sparse.csr_matrix,
                 use_pmap=False,
                 sharding: jax.sharding.PositionalSharding=None,
                 ):
    "Make a SpMV kernel w/ generic scatter-gather operations."
    import jax.numpy as np
    nrow = A.shape[0]
    # TODO consider conversion to COO to avoid custom code here
    col = np.array(A.indices)
    # numpy is faster than jax for this
    row = np.array(numpy.concatenate([
        i * numpy.ones(n, 'i')
        for i, n in enumerate(numpy.diff(A.indptr))]))
    dat = np.array(A.data)
    row, col = [_.astype('i') for _ in (row, col)]
    if (sharding is None) and use_pmap:
        col_ = _pad_up(col).reshape((vb.cores, -1))
        row_ = _pad_up(row).reshape((vb.cores, -1))
        dat_ = _pad_up(dat).reshape((vb.cores, -1))
        out_ = np.zeros((vb.cores, nrow))
        def part(c, r, d, o, x):
            # xc = x[c]
            # oatr = o.at[r]
            # return oatr.add(d * xc)
            return o.at[r].add(d * x[c])
        def spmv(x):
            f = jax.pmap(part)
            x_ = np.tile(x, (vb.cores, 1))
            outs = f(col_, row_, dat_, out_, x_)
            out = np.sum(outs, axis=0)
            return out
    elif sharding is not None:
        put = lambda x: jax.device_put(
            _pad_up(x).reshape((vb.cores, -1)), sharding)
        col_ = put(col)
        row_ = put(row)
        dat_ = put(dat)

        def spmv(x):
            out_ = jax.device_put(np.zeros((vb.cores, nrow)),
                                  sharding)
            f = jax.vmap(lambda c,r,d,o,x: o.at[r].add(d * x[c]))
            outs = f(col_, row_, dat_, out_, x)
            out = np.sum(outs, axis=0)
            return out
    else:
        def spmv(x):
            return np.zeros(nrow).at[row].add(dat * x[col])
    return spmv