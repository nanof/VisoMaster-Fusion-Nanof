#!/usr/bin/env python3
"""Imprime SHA256 de un fichero local (para rellenar hash en models_data)."""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys


def main() -> int:
    p = argparse.ArgumentParser(description="SHA256 de un modelo u otro fichero.")
    p.add_argument("path", type=pathlib.Path, help="Ruta al fichero")
    args = p.parse_args()
    path = args.path.resolve()
    if not path.is_file():
        print(f"Error: no existe o no es fichero: {path}", file=sys.stderr)
        return 1
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    print(h)
    print(f"size_bytes={path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
