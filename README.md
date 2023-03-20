# `vbjax`

`vbjax` is a Jax-based package for working with virtual brain style models.

Installs with `pip install vbjax`, or manually install dependencies with
`pip install numpy scipy jax jaxlib` or `conda install -y numpy scipy jax`.

## Examples



## Development
```
git clone https://github.com/ins-amu/vbjax
cd vbjax
pip install '.[dev]'
pytest
```

## Releases
a release of version `v1.2.3` requires following steps
- [ ] `git checkout main` tag releases from main for now
- [ ] edit `_version.py` to have correct release number
- [ ] `python -m vbjax._version tag` to create and push new tag
- [ ] `pip install '.[dev]'` ensures twine & build tools available
- [ ] `python -m build` builds wheels & source dists
- [ ] `twine upload dist/*` uploads to PyPI
- [ ] use GitHub UI to create new release
