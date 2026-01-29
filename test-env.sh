#!/bin/bash

set -eu

# tests a new env set up
deactivate || true
rm -rf test-env
uv venv test-env
source test-env/bin/activate
uv pip install -e ".[dev]"
pytest
deactivate
