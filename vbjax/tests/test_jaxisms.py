"""
These are tests of our understanding of how Jax works.  If these break
then probably things elsewhere will also break.

"""

import jax
import jax.test_util

def test_custom_vjp_simple():

    @jax.custom_vjp
    def foo(a, b):
        return 2*a+b, 3*a+0.5*b

    def foo_fwd(a, b):
        return foo(a, b), None

    def foo_bwd(res, g):
        g_1, g_2 = g
        return (2*g[0]+3*g[1], g[0] + 0.5*g[1])

    foo.defvjp(foo_fwd, foo_bwd)

    def bar(a, b):
        c, d = foo(a, b)
        return 2*c + 3*d

    jax.grad(bar, [0,1])(3.,4.)

    jax.test_util.check_grads(bar, (3.0, 4.0), order=1, modes=('rev',))
