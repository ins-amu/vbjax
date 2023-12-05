.. vbjax documentation master file, created by
   sphinx-quickstart on Tue Dec  5 11:11:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********************************
Welcome to vbjax's documentation!
***********************************

Introduction
############
`vbjax` is a Jax-based package for working with virtual brain style models.

Installation
############

Installs with `pip install "vbjax"`, but you can use the source,

.. code-block:: bash

   git clone https://github.com/ins-amu/vbjax
   cd vbjax
   pip install .[dev]


The primary additional dependency of vbjax is
[JAX](github.com/google/jax), which itself depends only on
NumPy, SciPy & opt-einsum, so it should be safe to add to your
existing projects.

gee pee you
------------

**CUDA**

If you have a CUDA-enabled GPU, you install the requisite dependencies like so

.. code-block:: bash

   pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


**M1/M2** 🍎

On newer Apple machines w/ M1 or M2 GPUs, JAX supports using the GPU experimentally
by installing just two extra packages:

.. code-block:: bash

   pip install ml-dtypes==0.2.0 jax-metal


About a third of vbjax tests fail due to absence of certain operations like n-dim
scatter/gather & FFTs, and it may not be faster because these CPUs already have
excellent memory bandwidth & latency hiding.

**CUDA** 🐳 

*BUT* because GPU software stack versions make aligning stars look like child's play,
container images are available and auto-built w/
[GitHub Actions](.github/workflows/docker-image.yml), so you can use w/ Docker

.. code-block:: bash

   docker run --rm -it ghcr.io/ins-amu/vbjax:main python3 -c 'import vbjax; print(vbjax.__version__)'


The images are built on Nvidia runtime images, so `--gpus all` is enough
for Jax to discover the GPU(s).


.. toctree::
   :maxdepth: 2
   :caption: Contents:


Tutorial
########

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Modules
########

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   
