"""
Functions for building time stepping loops.

"""

import jax
import jax.tree_util
import jax.numpy as np

zero = 0
tmap = jax.tree_util.tree_map

def heun_step(x, dfun, dt, *args, add=zero, adhoc=None, return_euler=False):
    """Use a Heun scheme to step state with a right hand sides dfun(.)
    and additional forcing term add.
    """
    adhoc = adhoc or (lambda x,*args: x)
    d1 = dfun(x, *args)
    if add is not zero:
        xi = tmap(lambda x,d,a: x + dt*d + a, x, d1, add)
    else:
        xi = tmap(lambda x,d: x + dt*d, x, d1)
    xi = adhoc(xi, *args)
    d2 = dfun(xi, *args)
    if add is not zero:
        nx = tmap(lambda x,d1,d2,a: x + dt*0.5*(d1 + d2) + a, x, d1, d2, add)
    else:
        nx = tmap(lambda x,d1,d2: x + dt*0.5*(d1 + d2), x, d1, d2)
    nx = adhoc(nx, *args)
    if return_euler:
        return xi, nx
    return nx


def _compute_noise(gfun, x, p, sqrt_dt, z_t):
    g = gfun(x, p)
    try: # maybe g & z_t are just arrays
        noise = g * sqrt_dt * z_t
    except TypeError: # one of them is a pytree
        if isinstance(g, float): # z_t is a pytree, g is a scalar
            noise = tmap(lambda z: g * sqrt_dt * z, z_t)
        # otherwise, both must be pytrees and they must match
        elif not jax.tree_util.tree_all(jax.tree_util.tree_structure(g) == 
                                        jax.tree_util.tree_structure(z_t)):
            raise ValueError("gfun and z_t must have the same pytree structure.")
        else:
            noise = tmap(lambda g,z: g * sqrt_dt * z, g, z_t)
    return noise

def make_sde(dt, dfun, gfun, adhoc=None, return_euler=False, unroll=10):
    """Use a stochastic Heun scheme to integrate autonomous stochastic
    differential equations (SDEs).

    Parameters
    ==========
    dt : float
        Time step
    dfun : function
        Function of the form `dfun(x, p)` that computes drift coefficients of
        the stochastic differential equation.
    gfun : function or float
        Function of the form `gfun(x, p)` that computes diffusion coefficients
        of the stochastic differential equation. If a numerical value is
        provided, this is used as a constant diffusion coefficient for additive
        linear SDE.
    adhoc : function or None
        Function of the form `f(x, p)` that allows making adhoc corrections
        to states after a step.
    return_euler: bool, default False
        Return solution with local Euler estimates.
    unroll: int, default 10
        Force unrolls the time stepping loop.

    Returns
    =======
    step : function
        Function of the form `step(x, z_t, p)` that takes one step in time
        according to the Heun scheme.
    loop : function
        Function of the form `loop(x0, zs, p)` that iteratively calls `step`
        for all `z`.

    Notes
    =====

    In both cases, a Jax compatible parameter set `p` is provided, either an array
    or some pytree compatible structure.

    Note that the integrator does not sample normally distributed noise, so this
    must be provided by the user.


    >>> import vbjax as vb
    >>> _, sde = vb.make_sde(1.0, lambda x, p: -x, 0.1)
    >>> sde(1.0, vb.randn(4), None)
    Array([ 0.5093468 ,  0.30794007,  0.07600437, -0.03876263], dtype=float32)

    """

    sqrt_dt = np.sqrt(dt)

    # gfun is a numerical value or a function f(x,p) -> sig
    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    def step(x, z_t, p):
        noise = _compute_noise(gfun, x, p, sqrt_dt, z_t)
        return heun_step(
            x, dfun, dt, p, add=noise, adhoc=adhoc,
            return_euler=return_euler)

    @jax.jit
    def loop(x0, zs, p):
        def op(x, z):
            x = step(x, z, p)
            # XXX gets unwieldy, how to improve?
            if return_euler:
                ex, x = x
            else:
                ex = None
            return x, (ex, x)
        _, xs = jax.lax.scan(op, x0, zs, unroll=unroll)
        if not return_euler:
            _, xs = xs
        return xs

    return step, loop


def make_ode(dt, dfun, adhoc=None):
    """Use a Heun scheme to integrate autonomous ordinary differential
    equations (ODEs).

    Parameters
    ==========
    dt : float
        Time step
    dfun : function
        Function of the form `dfun(x, p)` that computes derivatives of the
        ordinary differential equations.
    adhoc : function or None
        Function of the form `f(x, p)` that allows making adhoc corrections
        to states after a step.

    Returns
    =======
    step : function
        Function of the form `step(x, t, p)` that takes one step in time
        according to the Heun scheme.
    loop : function
        Function of the form `loop(x0, ts, p)` that iteratively calls `step`
        for all time steps `ts`.

    Notes
    =====

    In both cases, a Jax compatible parameter set `p` is provided, either an array
    or some pytree compatible structure.

    >>> import vbjax as vb, jax.numpy as np
    >>> _, ode = vb.make_ode(1.0, lambda x, p: -x)
    >>> ode(1.0, np.r_[:4], None)
    Array([0.5   , 0.25  , 0.125 , 0.0625], dtype=float32, weak_type=True)

    """

    def step(x, t, p):
        return heun_step(x, dfun, dt, p, adhoc=adhoc)

    @jax.jit
    def loop(x0, ts, p):
        def op(x, t):
            x = step(x, t, p)
            return x, x
        return jax.lax.scan(op, x0, ts)[1]

    return step, loop


def make_dde(dt, nh, dfun, unroll=10, adhoc=None):
    "Invokes make_sdde w/ gfun 0."
    return make_sdde(dt, nh, dfun, 0, unroll, adhoc=adhoc)


def make_sdde(dt, nh, dfun, gfun, unroll=1, zero_delays=False, adhoc=None):
    """Use a stochastic Heun scheme to integrate autonomous
    stochastic delay differential equations (SDEs).

    Parameters
    ==========
    dt : float
        Time step
    nh : int
        Maximum delay in time steps.
    dfun : function
        Function of the form `dfun(xt, x, t, p)` that computes drift coefficients of
        the stochastic differential equation.
    gfun : function or float
        Function of the form `gfun(x, p)` that computes diffusion coefficients
        of the stochastic differential equation. If a numerical value is
        provided, this is used as a constant diffusion coefficient for additive
        linear SDE.
    adhoc : function or None
        Function of the form `f(x,p)` that allows making adhoc corrections after
        each step.

    Returns
    =======
    step : function
        Function of the form `step((x_t,t), z_t, p)` that takes one step in time
        according to the Heun scheme.
    loop : function
        Function of the form `loop((xs, t), p)` that iteratively calls `step`
        for each `xs[nh:]` and starting time index `t`.

    Notes
    =====

    - A Jax compatible parameter set `p` is provided, either an array
      or some pytree compatible structure.
    - The integrator does not sample normally distributed noise, so this
      must be provided by the user.
    - The history buffer passed to the user functions, on the corrector
      stage of the Heun method, does not contain the predictor stage, for
      performance reasons, unless `zero_delays` is set to `True`.
      A good compromise can be to set all zero delays to dt.

    >>> import vbjax as vb, jax.numpy as np
    >>> _, sdde = vb.make_sdde(1.0, 2, lambda xt, x, t, p: -xt[t-2], 0.0)
    >>> x,t = sdde(np.ones(6)+10, None)
    >>> x
    Array([ 11.,  11.,  11.,   0., -11., -22.], dtype=float32)

    """

    heun = True
    sqrt_dt = np.sqrt(dt)
    nh = int(nh)

    # TODO nh == 0: return make_sde(dt, dfun, gfun) etc

    # gfun is a numerical value or a function f(x,p) -> sig
    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    if adhoc is None:
        adhoc = lambda x,p : x

    def step(buf_t, z_t, p):
        buf, t = buf_t
        x = tmap(lambda buf: buf[nh + t], buf)
        noise = _compute_noise(gfun, x, p, sqrt_dt, z_t)
        d1 = dfun(buf, x, nh + t, p)
        xi = tmap(lambda x,d,n: x + dt*d + n, x, d1, noise)
        xi = adhoc(xi, p)
        if heun:
            if zero_delays:
                # severe performance hit (5x+)
                buf = tmap(lambda buf, xi: buf.at[nh + t + 1].set(xi), buf, xi)
            d2 = dfun(buf, xi, nh + t + 1, p)
            nx = tmap(lambda x,d1,d2,n: x + dt*0.5*(d1 + d2) + n, x, d1, d2, noise)
            nx = adhoc(nx, p)
        else:
            nx = xi
        # jax.debug.print("buf len is {b}, going to write to {i}", b=buf.shape[0], i=nh+t+1)
        buf = tmap(lambda buf, nx: buf.at[nh + t + 1].set(nx), buf, nx)
        return (buf, t+1), nx

    # @jax.jit
    def loop(buf, p, t=0):
        "xt is the buffer, zt is (ts, zs), p is parameters."
        op = lambda xt, tz: step(xt, tz, p)
        dWt = tmap(lambda b: b[nh+1:], buf) # history is buf[nh:], current state is buf[nh], randn samples are buf[nh+1:]
        (buf, _), nxs = jax.lax.scan(op, (buf, t), dWt, unroll=unroll)
        return buf, nxs

    return step, loop


def make_continuation(run_chunk, chunk_len, max_lag, n_from, n_svar, stochastic=True):
    """
    Helper function to lower memory usage for longer simulations with time delays.
    WIP

    Takes a function

        run_chunk(buf, params) -> (buf, chunk_states)

    and returns another

        continue_chunk(buf, params, rng_key) -> (buf, chunk_states)

    The continue_chunk function wraps run_chunk and manages
    moving the latest states to the first part of buf and filling
    the rest with samples from N(0,1) if required.

    """
    from vbjax import randn

    # need to be compile time constants for dynamic_*
    i0 = chunk_len
    i1 = max_lag + 1

    @jax.jit
    def continue_chunk(buf, p, key):
        get, set = jax.lax.dynamic_slice, jax.lax.dynamic_update_slice
        buf = np.roll(buf, -i0, axis=0)

        # now fill the rest of the buffer with N(0,1) samples if stochastic
        # buf = buf.at[max_lag+1:].set( vb.randn(chunk_len-1, 2, n_from, key=key) )
        if stochastic:
            fill_val = randn(i0,n_svar,n_from, key=key)
            buf = set(buf, fill_val, (i1,0,0))
        else:
            # leave the buf since gfun() returns zero
            pass

        buf, rv = run_chunk(buf, p)
        return buf, rv
    return continue_chunk
