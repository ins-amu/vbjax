# Accelerated inference on fields: virtual brains in Jax

basic plan

- implement shtns like api + custom kernels for shtlc
  - focus on kernel design parity w/ sparse matrix
- epi2d, heun, obs2d, scan loop in time
- explore batching in time for hybrid

testing different methods, Jax allows

- numpyro: HMC & VI
- batched eval: efficient parameters sweeps for SBI
- run notebooks as tests with jupyter execute via dockerfile + gh actions

## setup

### conda

The primary dependency `jax` is easily installable with conda, e.g.

```
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p $SCRATCH/conda-vbjax -b
. $SCRATCH/conda-vbjax/bin/activate
conda install -y -q jax
pip install vbjax
```

### Dockerfile

See `cpu.dockerfile` or `gpu.dockerfile`.

### pipenv

pipenv is used to manage the dependencies, so you can install deps in a virtualenv 
and run Jupyter notebook like so:
```
pip install -U pipenv
pipenv install -d
pipenv run jupyter notebook
```

### installing shtns on mac

In the Python environment,
```
brew install fftw
git clone https://bitbucket.org/nschaeff/shtns
./configure --enable-python --disable-simd --prefix=/opt/homebrew
make -j && make install && python setup.py install
```

### Pipenv

Some packages that may be used are not in the Pipfile because pipenv
doesn't manage to install them.

- numba
- tvb-data & tvb-library
