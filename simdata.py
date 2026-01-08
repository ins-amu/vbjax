import numpy as np
import ctypes as ct
import scipy.sparse


class Sim(ct.Structure):
    _fields_ = [
        ('rng_seed', ct.c_int32),
        ('num_item', ct.c_int32),
        ('num_node', ct.c_int32),
        ('num_svar', ct.c_int32),
        ('num_time', ct.c_int32),
        ('num_params', ct.c_int32),
        ('num_spatial_params', ct.c_int32),
        ('num_simd', ct.c_int32),
        ('num_batch', ct.c_int32),
        ('dt', ct.c_float),
        ('num_skip', ct.c_int32),
        ('state_trace', ct.POINTER(ct.c_float)),
        ('z_scale', ct.POINTER(ct.c_float)),
        ('states', ct.POINTER(ct.c_float)),
        ('horizon', ct.c_int32),
        ('horizon_minus_1', ct.c_int32),
        ('horizon_is_pow_of_2', ct.c_bool),
        ('delay_buffer', ct.POINTER(ct.c_float)),
        ('params', ct.POINTER(ct.c_float)),
        ('spatial_params', ct.POINTER(ct.c_float)),
        ('num_nonzero', ct.c_int32),
        ('weights', ct.POINTER(ct.c_float)),
        ('indices', ct.POINTER(ct.c_int32)),
        ('indptr', ct.POINTER(ct.c_int32)),
        ('idelays', ct.POINTER(ct.c_int32)),
    ]


def map_array(sim, key, array):
    for k, v in sim._fields_:
        if k == key:
            break
    setattr(sim, key, array.ctypes.data_as(v))


def make_sim(
    csr_weights: scipy.sparse.csr_matrix,
    idelays: np.ndarray,
    sim_params: np.ndarray,
    z_scale: np.ndarray,
    horizon: int,
    rng_seed=43, num_item=8, num_node=90, num_svar=2, num_time=1000, dt=0.1,
    num_skip=5, num_simd=8
):
    sim = Sim()
    sim.rng_seed = rng_seed
    sim.num_item = num_item
    sim.num_simd = num_simd
    sim.num_batch = num_item // num_simd
    sim.num_node = num_node
    sim.num_svar = num_svar
    sim.num_time = num_time
    sim.dt = dt
    sim.num_skip = num_skip
    sim.horizon = 256
    assert num_item >= num_simd
    assert sim.num_batch*num_simd == num_item
    sim.horizon_minus_1 = sim.horizon - 1
    sim.horizon_is_pow_of_2 = True
    sim.num_params = sim_params.shape[0]
    sim.num_spatial_params = 0
    sim.num_nonzero = csr_weights.nnz

    sim_arrays = []

    def zeros(shape, dtype='f'):
        arr = np.zeros(shape, dtype)
        sim_arrays.append(arr)
        return arr

    sim_state_trace = zeros((sim.num_time // sim.num_skip + 2,
                             sim.num_svar,
                             sim.num_node,
                             sim.num_item), 'f')
    map_array(sim, 'state_trace', sim_state_trace)

    # TODO make this varying per item as well
    # sim_z_scale = (np.r_[0.01, 0.1]*np.sqrt(sim.dt)).astype('f')
    sim_z_scale = z_scale.astype('f')
    map_array(sim, 'z_scale', sim_z_scale)

    sim_states = zeros((sim.num_svar, sim.num_node, sim.num_item), 'f')
    map_array(sim, 'states', sim_states)

    # XXX needs to be power of 2
    sim_delay_buffer = zeros((sim.num_node, sim.horizon, sim.num_item), 'f')
    map_array(sim, 'delay_buffer', sim_delay_buffer)

    # sim_params = np.zeros((sim.num_params, sim.num_item), 'f')
    # sim_params[0] = 1.05;
    # sim_params[1] = 3.0;
    assert sim_params.shape == (sim.num_params, sim.num_item)
    sim_params = sim_params.copy().astype('f')
    sim_spatial_params = zeros((sim.num_spatial_params,
                                sim.num_node,
                                sim.num_item), 'f')
    map_array(sim, 'params', sim_params)
    map_array(sim, 'spatial_params', sim_spatial_params)

    sim_weights = zeros((sim.num_nonzero,), 'f')
    sim_indices = zeros((sim.num_nonzero,), np.int32)
    sim_indptr = zeros((sim.num_node+1,), np.int32)
    sim_idelays = zeros((sim.num_nonzero,), np.int32)

    sim_weights[:] = csr_weights.data.astype('f')
    sim_indices[:] = csr_weights.indices.astype('i')
    sim_indptr[:] = csr_weights.indptr.astype('i')
    sim_idelays[:] = idelays.astype('i')

    map_array(sim, 'weights', sim_weights)
    map_array(sim, 'indices', sim_indices)
    map_array(sim, 'indptr', sim_indptr)
    map_array(sim, 'idelays', sim_idelays)

    return sim, sim_arrays
