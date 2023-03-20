import numpy as np
import vbjax


def test_embed_polynomial():
    X = x, y, z = np.random.randn(3, 100)
    basis = vbjax.embed.embed_polynomial(X, max_order=2)
    assert basis.shape == (10, 100)

    Y = np.c_[x - x**2, z + x*y].T
    basis, coef = vbjax.embed.embed_polynomial(X, Y, max_order=2)
    assert basis.shape == (10, 100)
    assert coef.shape == (10,2)
    assert np.square(coef.T @ basis - Y).sum() < 1e-6


if __name__ == '__main__':
    test_embed_polynomial()
