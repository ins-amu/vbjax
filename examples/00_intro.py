#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
**Example 1**: A simple network of coupled Montbrio model nodes.

Starting with a few imports

.. literalinclude:: ../examples/00_intro.py
    :start-after: example-st\u0061rt
    :lines: 1-4
    :caption:

This example shows how to use the `vbjax` library to simulate a network of
Montbrio model nodes. The network is defined by the function `network` which
takes as arguments the state of the network and the parameters of the model.
The function returns the time derivative of the state of the network. 

.. literalinclude:: ../examples/00_intro.py
    :start-after: example-st\u0061rt
    :lines: 6-8

The function `make_sde` is used to create a function `loop` that simulates the
network for a given time interval and a given set of initial conditions.

.. literalinclude:: ../examples/00_intro.py
    :start-after: example-st\u0061rt
    :lines: 10

The function `vb.randn` is used to generate a set of noise samples. 
The dimesions are `(time, state, node)`. The first noise sample is used as the
initial condition of the network. The remaining noise samples are used to
generate the noise term of the stochastic differential equation.

.. literalinclude:: ../examples/00_intro.py
    :start-after: example-st\u0061rt
    :lines: 11

The function `loop` takes as arguments the initial conditions of the network,
vector of noise samples, and the parameters of the model. The function returns
the state of the network at each time step.

.. literalinclude:: ../examples/00_intro.py
    :start-after: example-st\u0061rt
    :lines: 12

The function `vb.plot_states` is used to plot the state of the network. The
function takes as arguments the state of the network, the format of the plot,
and the name of the output file.

.. literalinclude:: ../examples/00_intro.py
    :start-after: example-st\u0061rt
    :lines: 13

.. figure:: ../examples/images/example1.jpg
    :scale: 75 %


"""
# example-start
import os
import vbjax as vb
import jax.numpy as np
os.makedirs('images', exist_ok=True)

def network(x, p):
    c = 0.03*x.sum(axis=1)
    return vb.mpr_dfun(x, c, p)

_, loop = vb.make_sde(dt=0.01, dfun=network, gfun=0.1)
zs = vb.randn(500, 2, 32)
xs = loop(zs[0], zs[1:], vb.mpr_default_theta)
vb.plot_states(xs, 'rV', jpg='images/example1', show=False)

# example-end