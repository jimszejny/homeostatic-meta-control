"""Shared paths, grids and CLI for the experiment scripts."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from offledger import jsonio

ROOT = Path(__file__).resolve().parents[2]          # /.../paper
RESULTS_DIR = Path(os.environ.get("OFFLEDGER_RESULTS", ROOT / "results"))

OMEGAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]               # E1, E3, E6
OMEGAS_OSG = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]  # E4 (up to 64)
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMMA = 0.95


def main(name: str, run):
    ap = argparse.ArgumentParser(description=f"offledger experiment {name}")
    ap.add_argument("--out", default=str(RESULTS_DIR), help="results directory")
    args = ap.parse_args()
    t0 = time.perf_counter()
    res = run()
    path = jsonio.write(Path(args.out) / f"{name}.json", res)
    print(f"{name}: wrote {path} in {time.perf_counter() - t0:.1f} s")
