"""
Functions for region mapping.

"""

import jax.numpy as np


def make_region_mapping(region_map):
    """Create functions, for a given region mapping, which generate
    the region average from a surface vector or expand a regional vector
    to a surface vector.
    """

    def to_surface(x_region):
        return x_region[region_map]

    vtx_count = np.bincount(region_map)
    def to_region(x_surface):
        x_region = np.zeros(vtx_count.size)
        x_region_sum = x_region.at[region_map].add(x_surface)
        return x_region_sum / vtx_count

    return to_surface, to_region
