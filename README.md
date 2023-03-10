## `vbjax`

`vbjax` is a Jax-based package for working with virtual brain style models.

### conda

The primary dependency `jax` is easily installable with conda, e.g.

```
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -p $SCRATCH/conda-vbjax -b
. $SCRATCH/conda-vbjax/bin/activate
conda install -y -q jax
pip install vbjax
```

