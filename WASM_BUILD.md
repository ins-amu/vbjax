# WebAssembly Build for vbjax

This directory contains configuration for building vbjax for WebAssembly environments using emscripten-forge.

## Files

- `environment-wasm-build.yml`: Conda environment specification for building vbjax in emscripten
- `.github/workflows/emscripten-forge-build.yml`: GitHub Action workflow for automated builds

## Current Limitations

⚠️ **Important**: JAX and JAXlib do not currently have full support for WebAssembly/emscripten environments. This build creates the package structure but with limited functionality.

The WebAssembly build:
- ✅ Can be packaged and distributed via emscripten-forge
- ✅ Can be installed in xeus-python kernel environments
- ❌ Will have limited or no functionality due to JAX dependencies
- ❌ Cannot run the full vbjax neural mass models

## Future Work

As JAX support for WebAssembly/emscripten improves, this build configuration can be updated to provide full functionality.

## Building Locally

To build locally using pixi (recommended):

```bash
# Install pixi (see https://pixi.sh/latest/#installation)
# Then build the package
pixi run setup
pixi run build-emscripten-wasm32-pkg .
```

Or using rattler-build directly:

```bash
# Create environment
micromamba create -n emscripten-forge -f environment-wasm-build.yml
micromamba activate emscripten-forge

# Build package
rattler-build build \
  --recipe recipe.yaml \
  --target-platform=emscripten-wasm32 \
  -c https://repo.prefix.dev/emscripten-forge-dev \
  -c conda-forge
```

## Integration with xeus-python

Once built and published to emscripten-forge, vbjax can be installed in xeus-python kernels by adding it to the environment specification for JupyterLite deployments.