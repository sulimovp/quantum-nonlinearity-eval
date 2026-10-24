#!/usr/bin/env python3
"""
Copy canonical ``results/*.json`` into ``results/sweeps/n{n}_L{L}/seed_{seed}/``
so a multi-seed aggregation can treat the current reference run as one sweep arm
without re-executing the full pipeline for seed 42.

Usage (from repo root)::

  python scripts/mirror_protocol_results_to_sweep.py --seed 42 --n 3 --l 3
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# Files required for paper export / aggregation (subset of full pipeline).
MIRROR_JSON: tuple[str, ...] = (
    "nlc_scores.json",
    "nlc_detail.json",
    "nlc_metric_meta.json",
    "bppi_scores.json",
    "bppi_detail.json",
    "nss_scores.json",
    "nss_detail.json",
    "r2_curves.json",
    "ter_scores.json",
    "ter_scores_meta.json",
    "lorenz_results.json",
    "pqk_baseline_metrics.json",
    "nlps_final_scores.json",
    "nlps_composite_meta.json",
    "ter_learning_curves.json",
    "baseline_scores.json",
    "real_data_scores.json",
    "real_data_ter_scores.json",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=3, dest="n_qubits")
    p.add_argument("--l", type=int, default=3, dest="n_layers")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = root / "results"
    sub = f"sweeps/n{args.n_qubits}_L{args.n_layers}/seed_{args.seed}"
    dst = src / sub
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []
    for name in MIRROR_JSON:
        sp = src / name
        if not sp.is_file():
            missing.append(name)
            continue
        shutil.copy2(sp, dst / name)
        copied += 1

    print(f"Mirrored {copied} files -> {dst.relative_to(root)}")
    if missing:
        print("Missing (skipped):", ", ".join(missing))
    return 0 if copied else 1


if __name__ == "__main__":
    raise SystemExit(main())
