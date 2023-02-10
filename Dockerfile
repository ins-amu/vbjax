FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
CMD python3 -c 'import jax; print(jax.numpy.zeros(32).device())'
