#!/usr/bin/env python3
"""
Orchestrate multi-seed sweeps for the paper: mirror seed 42, run additional seeds,
then aggregate LaTeX table fragments.

Typical usage (from repo root)::

  # Full: mirror 42, run seeds 43--46 at (n,L)=(3,3), export tables
  python scripts/multiseed_paper_pipeline.py

  # Only re-export tables from an existing sweep directory
  python scripts/multiseed_paper_pipeline.py --export-only

Environment is inherited; the subprocess uses the same ``jupyter`` as ``run_neurips_pipeline``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(py: list[str], cwd: Path, env: dict | None = None) -> int:
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.call([sys.executable, "-X", "utf8", *py], cwd=str(cwd), env=e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--l", type=int, default=3)
    ap.add_argument(
        "--extra-seeds",
        default="43,44,45,46",
        help="Seeds to run after mirroring 42 (comma-separated).",
    )
    ap.add_argument(
        "--export-only",
        action="store_true",
        help="Only run paper_export_multiseed_tables (no mirror, no sweep).",
    )
    ap.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Mirror + export only (useful if sweep already finished).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    extra = [int(x.strip()) for x in args.extra_seeds.split(",") if x.strip()]

    if not args.export_only:
        rc = _run(
            [
                str(root / "scripts" / "mirror_protocol_results_to_sweep.py"),
                "--seed",
                "42",
                "--n",
                str(args.n),
                "--l",
                str(args.l),
            ],
            root,
        )
        if rc != 0:
            return rc

        if not args.skip_sweep and extra:
            nl = [f"{args.n},{args.l}"]
            seeds_csv = ",".join(str(s) for s in extra)
            rc = _run(
                [
                    str(root / "scripts" / "run_qnlep_sweep.py"),
                    "--seeds",
                    seeds_csv,
                    "--nl",
                    *nl,
                ],
                root,
            )
            if rc != 0:
                return rc

    rc = _run(
        [
            str(root / "scripts" / "paper_export_multiseed_tables.py"),
            "--n",
            str(args.n),
            "--l",
            str(args.l),
        ],
        root,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
