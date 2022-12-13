FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y python3-pip python3-{mpi4py,numba,scipy,matplotlib,pytest} \
 && pip install autograd pytest
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN apt-get install -y git build-essential libfftw3-dev
RUN git clone https://bitbucket.org/nschaeff/shtns \
 && cd shtns \
 && ./configure --enable-python --disable-simd \
 && make -j4 \
 && make install \
 && python setup.py install --user

RUN mkdir src
ADD ./ src/
WORKDIR src

CMD python -m pytest tests.py

