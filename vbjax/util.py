import numpy
import jax
import jax.dlpack
import jax.numpy as jp


def to_np(x: jax.numpy.ndarray) -> numpy.ndarray:
    if jax.default_backend() == 'gpu':
        return numpy.array(x)
    return numpy.from_dlpack(x)


def to_jax(x: numpy.ndarray):
    "Move NumPy array to JAX via DLPack."
    x_dlp = x.__dlpack__()
    x_jax: jax.numpy.ndarray = jax.dlpack.from_dlpack(x_dlp)
    # if jax.default_backend() == 'gpu':
    #     x_jax = jax.device_put(x_jax)
    return x_jax


def tuple_meshgrid(tup):
    "Applies meshgrid to arrays in a named tuple."
    grid_keys = []
    grid_parts = []
    for key in tup._fields:
        val = getattr(tup, key)
        if hasattr(val, 'shape'):
            grid_keys.append(key)
            grid_parts.append(val)
    grid = jp.meshgrid(*grid_parts, indexing='ij')
    # print(grid_keys, grid[0].shape, grid[1].shape)
    replacements = {k: g for k, g in zip(grid_keys, grid)}
    return grid_keys, tup._replace(**replacements)


def tuple_ravel(tup):
    "Flatten arrays in fields of tuple."
    scalars = {}
    arrays = {}
    for key, val in tup._asdict().items():
        if hasattr(val, 'shape'):
            arrays[key] = val
        else:
            scalars[key] = val
    shape1 = next(iter(arrays.values())).shape
    for key, val in arrays.items():
        if val.shape != shape1:
            raise ValueError(f"Shape mismatch for {key}: {val.shape} != {shape1}")
    # ravel arrays
    for key, val in arrays.items():
        arrays[key] = val.ravel()
    # expand scalars
    for key, val in scalars.items():
        scalars[key] = jax.numpy.ones(shape1).ravel()*val
    # combine
    new_values = {}
    new_values.update(scalars)
    new_values.update(arrays)
    return shape1, tup._replace(**new_values)
                                      

def tuple_shard(tup, n):
    "Shard arrays in fields of tuple."

    from jax.sharding import PositionalSharding
    sharding = PositionalSharding(jax.devices()[:n]).reshape(n, 1)
    sharded_values = {}
    for key, val in tup._asdict().items():
        try:
            val_ = val.reshape((n, -1))
        except TypeError:
            msg = f'Param {key} has shape {val.shape} which cannot be reshaped to {n} cores.'
            raise ValueError(msg)
        val = jax.device_put(val_, sharding)
        sharded_values[key] = val
    return tup._replace(**sharded_values)