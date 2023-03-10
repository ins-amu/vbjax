# from https://jax.readthedocs.io/en/latest/developer.html

FROM ubuntu:jammy

RUN apt-get update \
 && apt-get install -y openjdk-11-jdk git build-essential zip unzip wget \
	python3-pip python3-numpy python3-wheel

WORKDIR /opt/bazel

RUN wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-dist.zip \
 && unzip bazel-6.1.0-dist.zip \
 && ./compile.sh \
 && mv output/bazel /usr/bin/bazel

WORKDIR /opt/
RUN git clone https://github.com/google/jax
WORKDIR /opt/jax
RUN python3 build/build.py \
 && pip3 install /opt/jax/dist/jaxlib-0.4.7-cp310-cp310-manylinux2014_aarch64.whl --force-reinstall \
 && python3 setup.py develop \
 && python3 -c 'import jax.numpy as np; print(np.zeros(8).sum())'
