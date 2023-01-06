"""
Functions for building time stepping loops.

"""

import jax
import jax.numpy as np


def make_sde(dt, dfun, gfun):
    """Use a stochastic Heun scheme to integrate `dfun` & `gfun` at a time step `dt`.

    - `dfun(x, p)` computes drift coefficients of the model
    - `gfun(x, p)` computes diffusion coefficients.

    As a shortcut, a numerical value can be passed as the `gfun` for additive SDE cases.

    This function constructs and returns two functions: step and loop.

    - `step(x, z_t, p)`: takes one step in time according to the Heun scheme. 
    - `loop(x0, zs, p)`: iteratively calls `step` for all `z`.

    In both cases, a Jax compatible parameter set `p` is provided, either an array
    or some pytree compatible structure.

    """

    sqrt_dt = np.sqrt(dt)

    # gfun is a numerical value or a function f(x,p) -> sig
    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    def step(x, z_t, p):
        noise = gfun(x, p) * sqrt_dt * z_t
        d1 = dfun(x, p)
        x1 = x + dt*d1 + noise
        d2 = dfun(x1, p)
        return x + dt*0.5*(d1 + d2) + noise

    @jax.jit
    def loop(x0, zs, p):
        def op(x, z):
            x = step(x, z, p)
            return x, x
        return jax.lax.scan(op, x0, zs)[1]

    return step, loop


def make_ode(dt, dfun):
    """Use a deterministic Heun scheme to integrate `dfun` at a time step `dt`.

    - `dfun(x, p)` computes derivatives of the model

    This function constructs and returns two functions: step and loop.

    - `step(x, t, p)`: takes one step in time according to the Heun scheme. 
    - `loop(x0, ts, p)`: iteratively calls `step` for all `z`.

    In both cases, a Jax compatible parameter set `p` is provided, either an array
    or some pytree compatible structure.

    """

    def step(x, t, p):
        d1 = dfun(x, p)
        x1 = x + dt*d1
        d2 = dfun(x1, p)
        return x + dt*0.5*(d1 + d2)

    @jax.jit
    def loop(x0, ts, p):
        def op(x, t):
            x = step(x, t, p)
            return x, x
        return jax.lax.scan(op, x0, ts)[1]

    return step, loop
