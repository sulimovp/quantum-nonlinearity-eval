#!/usr/bin/env python3
"""
Aggregate flat metric JSON files (method -> float or null) across multiple runs.

Example (after copying per-seed results under results/sweeps/seed_*/):

  python scripts/aggregate_metric_jsons.py ^
    --glob "results/sweeps/seed_*/nlc_scores.json" ^
    --out results/nlc_scores_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _num(v):
    if v is None:
        return math.nan
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return math.nan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--glob",
        required=True,
        help='Glob relative to cwd, e.g. results/sweeps/seed_*/nlc_scores.json',
    )
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    root = Path.cwd()
    paths = sorted(root.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    per_method: dict[str, list[float]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if k.startswith("_") or k == "meta":
                continue
            x = _num(v)
            per_method.setdefault(k, []).append(x)

    stats = {}
    for m, xs in sorted(per_method.items()):
        finite = [x for x in xs if not math.isnan(x)]
        if not finite:
            stats[m] = {"n": len(xs), "mean": None, "std": None}
            continue
        mu = sum(finite) / len(finite)
        if len(finite) > 1:
            var = sum((x - mu) ** 2 for x in finite) / (len(finite) - 1)
            sig = math.sqrt(var)
        else:
            sig = 0.0
        stats[m] = {"n": len(finite), "mean": mu, "std": sig, "n_nan": len(xs) - len(finite)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_glob": args.glob,
        "n_files": len(paths),
        "files": [str(x.relative_to(root)) for x in paths],
        "by_method": stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} ({len(paths)} files, {len(stats)} methods)")


if __name__ == "__main__":
    main()
