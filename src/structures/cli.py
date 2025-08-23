"""Command-line interface for the structures package.

This module exposes a simple CLI entrypoint for quick interactions. It also
serves as an example for docstring-based documentation via mkdocstrings.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .example import fibonacci


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Returns:
        An argparse.ArgumentParser configured for the CLI.
    """
    parser = argparse.ArgumentParser(prog="structures", description="Structures CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fib = subparsers.add_parser("fibonacci", help="Compute the n-th Fibonacci number")
    fib.add_argument("n", type=int, help="Index n (>= 0)")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entrypoint for the CLI.

    Args:
        argv: Optional iterable of arguments, defaults to sys.argv if None.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fibonacci":
        value = fibonacci(args.n)
        print(value)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
