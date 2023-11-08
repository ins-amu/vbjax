"""
Tests and benchmarks for a more generic uber loop.

- constant args
- time-dep args
- jit-time constants vs args
- monitors
- jax.checkpoint

"""

import jax
import jax.numpy as np
import vbjax as vb


def _heun_step(x, dt, f, args):
    f1 = f(x, args)
    f2 = f(x + dt*f1, args)
    return x + dt*0.5*(f1 + f2)

def _dfun1(xy, args):
    SC, a, k, stim = args
    x, y = xy
    c = np.dot(SC, x)*k
    dx = 5.0*(x - x*x*x/3 + y)
    dy = 0.2*(a - x + stim + c)
    return np.array([dx, dy])


def make_loop(dt, dfun, constants=()):
    def loop(initial_state, parameters=(), t_parameters=()):
        def step(state, t_parameters):
            args = constants + parameters + t_parameters
            next_state = _heun_step(state, dt, dfun, args)
            return next_state, next_state
        _, states = jax.lax.scan(step, initial_state, t_parameters)
        return states
    return loop


# can we go for the fully generic graph idea?
# stim(t) -> node -> monitor?
def make_loop_graph(steps, constants):
    def loop(initials, params, Tdeps):
        def body(states, tdeps):
            nexts = tuple(
                s(*args)
                for s, *args in zip(steps, constants, params, states, tdeps)
            )
            return nexts, nexts
        _, sol = jax.lax.scan(body, initials, Tdeps)
        return sol
    return loop
# would be more flexible if we had a dict or struct
# e.g. jax dataclasses

















