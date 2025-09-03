#!/usr/bin/env python3
"""Simple CLI entrypoint for the vbjax package.

Usage: python -m vbjax <subcommand> [args]

Currently implements:
 - run-tests [pytest args...]  : run the project's tests via pytest

This uses argparse so it's easy to extend later or swap to rich/typer.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m vbjax", description="vbjax convenience CLI")
    subs = p.add_subparsers(dest="command")
    subs.required = True

    p_tests = subs.add_parser("run-tests", help="Run tests via pytest")
    # capture remaining args and forward to pytest
    p_tests.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Arguments forwarded to pytest")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-tests":
        # argparse.REMAINDER will include any leading '--'; keep as-is
        rc = run_tests(args.pytest_args or [])
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
