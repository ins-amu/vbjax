Numerical Integration Guide
============================

This guide covers the numerical integration methods available in vbjax for solving:

- **Ordinary Differential Equations (ODEs)**
- **Stochastic Differential Equations (SDEs)**
- **Delay Differential Equations (DDEs)**
- **Stochastic Delay Differential Equations (SDDEs)**

Installation
------------

.. code-block:: bash

   pip install vbjax

Quick Start Examples
--------------------

Simple ODE
^^^^^^^^^^

Solve the exponential decay equation: dx/dt = -x

.. code-block:: python

   import jax.numpy as jnp
   from vbjax import make_ode

   # Define the differential equation: dx/dt = -x
   def dfun(x, p):
       return -x

   # Setup
   dt = 0.01
   t_max = 5.0
   ts = jnp.arange(0, t_max, dt)
   x0 = 1.0

   # Create integrator and solve
   step, loop = make_ode(dt, dfun, method='rk4')
   x = loop(x0, ts, None)

   print(f"x(0) = {x[0]:.4f}, x(T) = {x[-1]:.4f}")

System of ODEs: Harmonic Oscillator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def harmonic_oscillator(state, p):
       """state = [position, velocity]"""
       x, v = state
       return jnp.array([v, -x])  # [dx/dt, dv/dt]

   x0 = jnp.array([1.0, 0.0])
   step, loop = make_ode(dt, harmonic_oscillator, method='rk4')
   states = loop(x0, ts, None)

Stochastic Differential Equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ornstein-Uhlenbeck process with pre-generated noise:

.. code-block:: python

   from jax import random
   from vbjax import make_sde

   def drift(x, p):
       return -p * x  # Drift coefficient

   def diffusion(x, p):
       return 0.5  # Diffusion coefficient

   # Generate noise beforehand
   key = random.PRNGKey(42)
   n_steps = int(t_max / dt)
   zs = random.normal(key, (n_steps,))

   # Solve
   theta = 1.0
   step, loop = make_sde(dt, drift, diffusion)
   x = loop(x0=2.0, zs=zs, p=theta)

Delay Differential Equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from vbjax import make_dde

   def delayed_system(xt, x, t, p):
       """DDE: dx/dt = -x(t-τ)"""
       tau_steps = p
       x_delayed = xt[t - tau_steps]
       return -x_delayed

   # Setup with delay
   dt = 0.01
   tau = 1.0  # delay time
   tau_steps = int(tau / dt)
   n_steps = 100

   # Create history buffer
   buffer_size = tau_steps + 1 + n_steps
   history = jnp.ones(buffer_size)
   history = history.at[tau_steps + 1:].set(0.0)

   # Solve
   step, loop = make_dde(dt, tau_steps, delayed_system)
   buf_final, x = loop(history, tau_steps, t=0)

API Reference
-------------

make_ode
^^^^^^^^

Create an ordinary differential equation integrator.

.. function:: make_ode(dt, dfun, method='heun', adhoc=None)

   :param float dt: Time step size
   :param callable dfun: Function ``dfun(x, p)`` that returns dx/dt
   :param str method: Integration method - ``'euler'``, ``'heun'``, or ``'rk4'``
   :param callable adhoc: Optional function ``f(x, p)`` for post-step corrections
   :return: Tuple of (step, loop) functions
   :rtype: tuple

   **Returns:**

   - ``step``: Single step function ``step(x, t, p)``
   - ``loop``: Full integration function ``loop(x0, ts, p)``

   **Example:**

   .. code-block:: python

      def dfun(x, p):
          return -p * x  # dx/dt = -p*x

      step, loop = make_ode(dt=0.01, dfun=dfun, method='rk4')
      x = loop(x0=1.0, ts=jnp.arange(0, 5, 0.01), p=1.0)

make_sde
^^^^^^^^

Create a stochastic differential equation integrator using the stochastic Heun method.

.. function:: make_sde(dt, dfun, gfun, adhoc=None, return_euler=False, unroll=10)

   :param float dt: Time step size
   :param callable dfun: Drift function ``dfun(x, p)`` returning drift coefficient
   :param gfun: Diffusion function ``gfun(x, p)`` or constant diffusion
   :type gfun: callable or float
   :param callable adhoc: Optional post-step correction function
   :param bool return_euler: If True, also return Euler estimates
   :param int unroll: Loop unroll factor for performance
   :return: Tuple of (step, loop) functions
   :rtype: tuple

   **Returns:**

   - ``step``: Single step function ``step(x, z_t, p)``
   - ``loop``: Full integration function ``loop(x0, zs, p)`` where ``zs`` are noise samples

   **Example:**

   .. code-block:: python

      def drift(x, p):
          theta = p
          return -theta * x

      def diffusion(x, p):
          return 0.5

      # Generate noise
      key = random.PRNGKey(42)
      zs = random.normal(key, (n_steps,))

      step, loop = make_sde(dt=0.01, dfun=drift, gfun=diffusion)
      x = loop(x0=1.0, zs=zs, p=1.0)

make_dde
^^^^^^^^

Create a delay differential equation integrator.

.. function:: make_dde(dt, nh, dfun, unroll=10, adhoc=None)

   :param float dt: Time step size
   :param int nh: Maximum delay in time steps
   :param callable dfun: Function ``dfun(xt, x, t, p)`` where:

      - ``xt``: History buffer
      - ``x``: Current state
      - ``t``: Current time index in buffer
      - ``p``: Parameters

   :param int unroll: Loop unroll factor
   :param callable adhoc: Optional post-step correction
   :return: Tuple of (step, loop) functions
   :rtype: tuple

   **Example:**

   .. code-block:: python

      def dfun(xt, x, t, p):
          tau_steps = p
          return -xt[t - tau_steps]  # dx/dt = -x(t-τ)

      tau_steps = 100
      buffer_size = tau_steps + 1 + n_steps
      history = jnp.ones(buffer_size)

      step, loop = make_dde(dt=0.01, nh=tau_steps, dfun=dfun)
      buf, x = loop(history, tau_steps)

make_sdde
^^^^^^^^^

Create a stochastic delay differential equation integrator.

.. function:: make_sdde(dt, nh, dfun, gfun, unroll=1, zero_delays=False, adhoc=None)

   :param float dt: Time step size
   :param int nh: Maximum delay in time steps (set to 0 for no delay)
   :param callable dfun: Drift function ``dfun(xt, x, t, p)``
   :param gfun: Diffusion function or constant
   :type gfun: callable or float
   :param int unroll: Loop unroll factor
   :param bool zero_delays: Include predictor in history (performance vs accuracy)
   :param callable adhoc: Optional post-step correction
   :return: Tuple of (step, loop) functions
   :rtype: tuple

   .. note::
      When ``nh=0``, the function automatically uses the optimized SDE integrator
      for better performance and accuracy.

Integration Methods
-------------------

Accuracy Comparison
^^^^^^^^^^^^^^^^^^^

For a smooth ODE with step size dt:

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 15 45

   * - Method
     - Order
     - Error
     - Speed
     - Use Case
   * - Euler
     - 1
     - O(dt)
     - Fastest
     - Quick prototyping, very smooth problems
   * - Heun
     - 2
     - O(dt²)
     - Medium
     - Good balance, **default choice**
   * - RK4
     - 4
     - O(dt⁴)
     - Slower
     - High accuracy requirements

Method Selection Guide
^^^^^^^^^^^^^^^^^^^^^^

**Euler Method:**

- Fastest execution
- Use for very smooth problems or when speed is critical
- Good for quick prototyping

**Heun Method (default):**

- Best balance of speed and accuracy
- Recommended for most applications
- 2nd order accurate

**RK4 (Runge-Kutta 4th order):**

- Highest accuracy among available methods
- Use when precision is critical
- Required for stiff or complex dynamics

Comparison with SciPy
---------------------

Advantages of vbjax
^^^^^^^^^^^^^^^^^^^

- **10-100x faster** with JIT compilation
- GPU/TPU support
- Automatic differentiation through integrators
- Vectorization over multiple initial conditions using ``jax.vmap``

Advantages of SciPy
^^^^^^^^^^^^^^^^^^^

- Adaptive step size methods
- Better for stiff problems
- More mature error control
- Wider method selection (RK45, DOP853, etc.)

Recommendation
^^^^^^^^^^^^^^

- Use **vbjax** for speed-critical applications, parameter fitting (with gradients), or GPU acceleration
- Use **SciPy** for one-off integrations or stiff problems requiring adaptive methods

Accuracy Validation Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from scipy.integrate import solve_ivp
   from vbjax import make_ode

   def lorenz(x, p):
       sigma, rho, beta = p
       x1, x2, x3 = x
       return jnp.array([
           sigma * (x2 - x1),
           x1 * (rho - x3) - x2,
           x1 * x2 - beta * x3
       ])

   # JAX solution
   step, loop = make_ode(0.001, lorenz, method='rk4')
   x_jax = loop(x0, ts, params)

   # SciPy solution
   def lorenz_scipy(t, x):
       return np.array(lorenz(x, params))
   
   sol = solve_ivp(lorenz_scipy, [0, t_max], x0, 
                   t_eval=ts, method='RK45')
   x_scipy = sol.y.T

   # Compare
   diff = np.linalg.norm(x_jax - x_scipy, axis=1)
   print(f"Max difference: {np.max(diff):.2e}")
   # Typical: ~1e-6 to 1e-8

Advanced Usage
--------------

Automatic Differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute gradients through the integrator:

.. code-block:: python

   import jax

   def solve_ode(params):
       step, loop = make_ode(dt, lambda x, p: -p * x, method='rk4')
       return loop(x0, ts, params)

   # Gradient of final state w.r.t. parameters
   grad_fn = jax.grad(lambda p: solve_ode(p)[-1])
   gradient = grad_fn(1.0)

Vectorization Over Initial Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Solve for multiple initial conditions in parallel using ``jax.vmap``:

.. code-block:: python

   # Solve for multiple initial conditions in parallel
   x0_batch = jnp.array([1.0, 2.0, 3.0, 4.0])
   step, loop = make_ode(dt, dfun, method='rk4')

   # Vectorize over first argument (x0)
   loop_vmap = jax.vmap(loop, in_axes=(0, None, None))
   x_batch = loop_vmap(x0_batch, ts, params)
   # Shape: (4, len(ts)) for each initial condition

Custom Post-Step Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enforce constraints after each integration step:

.. code-block:: python

   def adhoc(x, p):
       """Enforce positivity constraint"""
       return jnp.maximum(x, 0.0)

   step, loop = make_ode(dt, dfun, method='rk4', adhoc=adhoc)

Performance Tips
----------------

1. **Use JIT compilation**: The ``loop`` functions are pre-compiled with ``@jax.jit``
2. **Batch initial conditions**: Use ``jax.vmap`` to solve multiple ICs in parallel
3. **GPU acceleration**: Set ``JAX_PLATFORM_NAME=gpu`` for GPU execution
4. **Unroll parameter**: Adjust ``unroll`` in SDEs/DDEs for better performance
5. **Memory**: For long simulations with delays, use ``make_continuation`` helper

Common Issues and Solutions
---------------------------

TracerError or ConcretizationError
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Error when using Python control flow or NumPy operations.

**Solution**: 

- Use JAX functions (``jnp`` not ``np``)
- Avoid Python control flow
- Use ``jax.lax.cond`` for conditionals

Numerical Instability
^^^^^^^^^^^^^^^^^^^^^

**Problem**: Solutions explode or become NaN.

**Solution**:

- Reduce ``dt`` (time step)
- Use higher-order method (RK4 instead of Euler)
- Check problem formulation

Slow First Run
^^^^^^^^^^^^^^

**Problem**: First execution is very slow.

**Solution**: 

- This is JIT compilation (normal behavior)
- Subsequent runs will be much faster
- Warmup with a dummy run if needed

Memory Error with DDEs
^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Out of memory for long delay simulations.

**Solution**:

- Use ``make_continuation`` for long simulations
- Reduce buffer size if possible
- Process data in chunks

Zero Delay Optimization
-----------------------

When using ``make_sdde`` with ``nh=0`` (no delay), the integrator automatically
switches to the optimized SDE implementation for better performance and accuracy.

.. code-block:: python

   # Automatically uses optimized SDE integrator when nh=0
   step, loop = make_sdde(dt=0.01, nh=0, dfun=dfun, gfun=0.0)
   
   # This is equivalent to and faster than the full SDDE machinery

This optimization is transparent to the user and ensures the best performance
for your specific use case.
