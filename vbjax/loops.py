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


def make_dde(dt, nh, dfun, unroll=10):
    """Use a deterministic Heun scheme to integrate `dfun` at a time step `dt`,
    with maximum time delay (in steps) provided by `nh`. If `nh` is zero, this
    function invokes `make_ode`.

    - `dfun(xt, x, t p)` computes derivatives of the model

    where

    - `xt` is a buffer of shape (nsvar, nh + ntime)
    - `x` is the current state variable vector
    - `t` is the current time index
    - `p` is the parameter set

    The extra arguments `xt` & `t` enable time delay implementations; see
    example notebooks.

    This function constructs and returns two functions: step and loop.

    - `step(x, t, p)`: takes one step in time according to the Heun scheme. 
    - `loop(xt, ts, p)`: iteratively calls `step` for all `z`.

    In both cases, a Jax compatible parameter set `p` is provided, either an array
    or some pytree compatible structure.

    Indexing out of bounds in Jax is undefined behavior, so if `nh` is not
    large enough, or the buffer `xt` is not long enough for the delays used
    by `dfun`, the result will be silently incorrect.

    """

    if nh == 0:
        print('nh==0: using make_ode to avoid overhead')
        dfun_ = lambda x, p: dfun(x.reshape((-1, 1)), x, 0, p)
        step, loop = make_ode(dt, dfun_)
        def loop_(xt, ts, p):
            return loop(xt[:,0], ts, p).T
        return step, loop_

    def step(xt, t, p):
        x = xt[:, nh + t]
        d1 = dfun(xt, x, t, p)
        xi = x + dt*d1
        xt = xt.at[:, nh + t + 1].set(xi)
        d2 = dfun(xt, xi, t+1, p)
        nx = x + dt*0.5*(d1 + d2)
        xt = xt.at[:, nh + t + 1].set(nx)
        return xt, nx

    @jax.jit
    def loop(xt, ts, p):
        op = lambda xt, t: step(xt, t, p)
        return jax.lax.scan(op, xt, ts, unroll=unroll)[0]

    return step, loop



def make_sdde(dt, nh, dfun, gfun, unroll=10):
    "Combines semantics of make_dde & make_sde; TODO docstring"

    sqrt_dt = np.sqrt(dt)

    # TODO nh == 0: return make_sde(dt, dfun, gfun) etc

    # gfun is a numerical value or a function f(x,p) -> sig
    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    def step(xt, zt, p):
        "xt is buffer, zt is (ts[i], zs[i]), p is parameters."
        t, z_t = zt
        noise = gfun(x, p) * sqrt_dt * z_t
        x = xt[:, nh + t]
        d1 = dfun(xt, x, t, p)
        xi = x + dt*d1 + noise
        xt = xt.at[:, nh + t + 1].set(xi)
        d2 = dfun(xt, xi, t+1, p)
        nx = x + dt*0.5*(d1 + d2) + noise
        xt = xt.at[:, nh + t + 1].set(nx)
        return xt, nx

    @jax.jit
    def loop(xt, zt, p):
        "xt is the buffer, zt is (ts, zs), p is parameters."
        op = lambda xt, tz: step(xt, zt, p)
        return jax.lax.scan(op, xt, zt, unroll=unroll)[0]

    return step, loop
