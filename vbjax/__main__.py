#!/usr/bin/env python3
"""Simple CLI entrypoint for the vbjax package.

Usage: python -m vbjax <subcommand> [args]
"""
from __future__ import annotations
import sys
from ._cli import main

if __name__ == "__main__":
    main()
