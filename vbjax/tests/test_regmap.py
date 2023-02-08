import jax.numpy as np

from vbjax import make_region_mapping


def test_region_mapping():
    rm = np.r_[0, 0, 0, 1, 1, 2]
    to_surface, to_region = make_region_mapping(rm)
    x_region = np.r_[0.3, 0.5, -2.0]
    assert (to_region(to_surface(x_region)) == x_region).all()
