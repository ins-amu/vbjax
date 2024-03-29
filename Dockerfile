FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update
RUN apt-get install -y python3-pip libopenmpi-dev openmpi-bin
RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install jupyterlab brian2 matplotlib joblib scipy vbjax mpi4jax numpyro
CMD python3 -c 'import jax; print(jax.numpy.zeros(32).device())'
