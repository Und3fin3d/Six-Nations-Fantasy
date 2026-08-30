#!/usr/bin/env python3
"""Reject model predictions outside the exact pinned Python environment."""

from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#]+)$")


def check(
    requirements: Path = Path("requirements-model.txt"),
    *, python_version: tuple[int, int] | None = None,
    installed_version=version,
) -> list[str]:
    errors = []
    current_python = sys.version_info[:2] if python_version is None else python_version
    if current_python != (3, 11):
        errors.append(
            f"Python 3.11 is required; found {current_python[0]}.{current_python[1]}"
        )
    for line in requirements.read_text().splitlines():
        match = PIN.match(line.strip())
        if not match:
            continue
        package, expected = match.groups()
        try:
            actual = installed_version(package)
        except PackageNotFoundError:
            errors.append(f"{package}=={expected} is required; package is missing")
            continue
        if actual != expected:
            errors.append(f"{package}=={expected} is required; found {actual}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    errors = check()
    if errors and not args.quiet:
        print("Model environment is incompatible:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
