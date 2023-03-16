import numpy as np
import vbjax


def test_make_dense_layers():
    """Test make_dense_layers."""
    for in_dim, extra_in, ld in [(10, 0, [10]), (10, 5, [10, 10])]:
        p, f = vbjax.make_dense_layers(in_dim, ld, extra_in=extra_in)
        x = np.random.randn(in_dim + extra_in, 1)
        y = f(p, x)
        assert y.shape == (in_dim, 1)


if __name__ == "__main__":
    test_make_dense()