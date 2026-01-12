import argparse
import subprocess
import sys
import importlib.util
import importlib
import os
import itertools
from typing import List, Optional, Any, Dict
import numpy as np

def run_tests(pytest_args: Optional[List[str]] = None) -> int:
    """Run pytest in a subprocess and return its exit code."""
    if pytest_args is None:
        pytest_args = []
    cmd = [sys.executable, "-m", "pytest"] + pytest_args
    # Forward exit code from pytest
    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        return 130

def load_module(path: str):
    """Load a python module from a file path or module name."""
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("sim_module", path)
        if spec is None or spec.loader is None:
             raise ImportError(f"Could not load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        # try as module
        try:
             return importlib.import_module(path)
        except ImportError:
             raise FileNotFoundError(f"File or module {path} not found.")

def parse_val(s):
    """Parse a string to int, float, bool or string."""
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def parse_slice(s):
    """Parse a slice string like 1:10:1 or 1:10."""
    parts = s.split(':')
    if len(parts) == 2:
        start = float(parts[0])
        stop = float(parts[1])
        step = 1.0
    elif len(parts) == 3:
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2])
    else:
        raise ValueError(f"Invalid slice format: {s}. Expected start:stop or start:stop:step")
    return np.arange(start, stop, step)

def run_sim_logic(module, extras):
    if not hasattr(module, 'run'):
        print(f"Error: Module does not have a 'run' function.")
        sys.exit(1)

    # Get default config if available
    config = getattr(module, 'default_config', {}).copy()

    # Parse extras
    for extra in extras:
        if '=' in extra:
            k, v = extra.split('=', 1)
            val = parse_val(v)
            config[k] = val
        else:
             print(f"Warning: Ignoring argument {extra}, format should be key=value")

    print(f"Running simulation with config: {config}")
    result = module.run(config)
    print("Result:", result)

def exec_sim(args, extras):
    module = load_module(args.file)
    run_sim_logic(module, extras)

def sweep_sim(args, extras):
    module = load_module(args.file)

    if not hasattr(module, 'run'):
        print(f"Error: Module {args.file} does not have a 'run' function.")
        sys.exit(1)

    config = getattr(module, 'default_config', {}).copy()

    sweep_params = {}
    fixed_params = {}

    # Parse extras
    for extra in extras:
        if '=' in extra:
            k, v = extra.split('=', 1)
            if ':' in v:
                 # sweep parameter
                 sweep_params[k] = parse_slice(v)
            else:
                 # fixed override
                 fixed_params[k] = parse_val(v)
                 config[k] = fixed_params[k]
        else:
             print(f"Warning: Ignoring argument {extra}, format should be key=value")

    if not sweep_params:
        print("No sweep parameters provided. Use start:stop:step for values.")
        # Reconstruct extras for run_sim_logic or just use config?
        # Since we already parsed extras into config, we can just pass that?
        # But run_sim_logic re-parses extras.
        # So we can just call run_sim_logic with the original extras.
        run_sim_logic(module, extras)
        return

    print(f"Sweeping {args.file}")
    print(f"Fixed config: {config}")
    print(f"Sweep params: {list(sweep_params.keys())}")

    keys = list(sweep_params.keys())
    values = list(sweep_params.values())

    # Create grid
    product = list(itertools.product(*values))
    print(f"Total combinations: {len(product)}")

    results = []
    configs = []

    for i, p_values in enumerate(product):
        current_config = config.copy()
        for k, v in zip(keys, p_values):
            current_config[k] = v
        # print progress every 10%
        if i % max(1, len(product) // 10) == 0:
             print(f"Step {i+1}/{len(product)}")

        configs.append(current_config)
        res = module.run(current_config)
        results.append(res)

    try:
        results_arr = np.array(results)
    except Exception:
        results_arr = np.array(results, dtype=object)

    output_file = args.output
    if not output_file:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        output_file = f"{base_name}_sweep.npz"

    print(f"Saving results to {output_file}")
    np.savez(output_file, results=results_arr, keys=keys, values=np.array(values, dtype=object), configs=np.array(configs, dtype=object))
    print("Sweep complete.")

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m vbjax", description="vbjax convenience CLI")
    subs = parser.add_subparsers(dest="command")
    subs.required = True

    # run-tests
    p_tests = subs.add_parser("run-tests", help="Run tests via pytest")
    p_tests.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Arguments forwarded to pytest")

    # exec
    p_exec = subs.add_parser("exec", help="Execute a simulation file")
    p_exec.add_argument("file", help="Path to python file or module")
    p_exec.add_argument("overrides", nargs='*', help="Configuration overrides (key=value)")

    # sweep
    p_sweep = subs.add_parser("sweep", help="Run a parameter sweep")
    p_sweep.add_argument("file", help="Path to python file or module")
    p_sweep.add_argument("--output", "-o", help="Output file path (default: <filename>_sweep.npz)")
    p_sweep.add_argument("overrides", nargs='*', help="Sweep parameters (key=start:stop:step) or fixed overrides (key=val)")


    args = parser.parse_args(argv)

    if args.command == "run-tests":
        rc = run_tests(args.pytest_args or [])
        sys.exit(rc)
    elif args.command == "exec":
        exec_sim(args, args.overrides)
    elif args.command == "sweep":
        sweep_sim(args, args.overrides)
