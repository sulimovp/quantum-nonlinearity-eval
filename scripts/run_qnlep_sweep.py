#!/usr/bin/env python3
"""
Run the full QNLEP notebook pipeline (03→04→07→08→11→09) for each (n, L) pair and each seed.

Artifacts are written under ``results/<QNLEP_RESULTS_SUBDIR>/``, with default::

    results/sweeps/n{n}_L{L}/seed_{seed}/

Environment variables set for each subprocess (see ``src/qnlep_run_paths.py``):

- ``QNLEP_SEED``
- ``QNLEP_N_QUBITS``, ``QNLEP_N_LAYERS``
- ``QNLEP_RESULTS_SUBDIR`` — relative to ``results/`` (no leading ``results/`` prefix)
- ``PYTHONUTF8``, ``PYTHONIOENCODING`` — for Windows notebook UTF-8 reads

After all runs, prints suggested ``aggregate_metric_jsons.py`` commands for mean±std
per (n, L) grid point.

If the environment variable ``QNLEP_SWEEP_FINALIZE_PAPER`` is set to ``1`` (or
``true``), also runs ``scripts/paper_export_multiseed_tables.py`` for each
``(n, L)`` in the sweep so ``neurips2026/generated/*.tex`` is refreshed.

Example::

  python scripts/run_qnlep_sweep.py --seeds 42,43,44 --nl 3,3 4,3 3,4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _infer_start_from(target_dir: Path) -> str | None:
    """Resume a partial sweep directory without re-running finished notebooks."""
    if (target_dir / "nlps_final_scores.json").is_file():
        return None
    if (target_dir / "ter_scores.json").is_file():
        return "11_comparison_metrics.ipynb"
    if (target_dir / "nss_scores.json").is_file():
        return "08_lorenz_attractor.ipynb"
    if (target_dir / "bppi_scores.json").is_file():
        return "07_nss_noise_stability.ipynb"
    if (target_dir / "nlc_scores.json").is_file():
        return "04_bppi_barren_plateau.ipynb"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46",
        help="Comma-separated integer seeds (default: five seeds).",
    )
    parser.add_argument(
        "--nl",
        nargs="+",
        default=["3,3"],
        metavar="n,L",
        help='Width/layer pairs, e.g. ``3,3`` ``4,3`` ``3,4`` (default: 3,3 only).',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs only.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip runs whose target directory already contains nlps_final_scores.json (default: on).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override --skip-existing and re-run every (n,L,seed) even if its directory looks complete.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not stop the whole sweep on a single-seed pipeline failure; record and continue.",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Forward --start-from to run_neurips_pipeline.py for every seed (e.g. 09_nlps_composite_score.ipynb to resume from notebook 09).",
    )
    parser.add_argument(
        "--through",
        type=str,
        default=None,
        help="Forward --through to run_neurips_pipeline.py (e.g. 09_nlps_composite_score.ipynb).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    nl_pairs: list[tuple[int, int]] = []
    for item in args.nl:
        parts = item.replace(" ", "").split(",")
        if len(parts) != 2:
            raise SystemExit(f"Bad --nl entry {item!r}; expected n,L")
        nl_pairs.append((int(parts[0]), int(parts[1])))

    runs = []
    for n, L in nl_pairs:
        for seed in seeds:
            sub = f"sweeps/n{n}_L{L}/seed_{seed}"
            runs.append((n, L, seed, sub))

    print(f"Planned {len(runs)} pipeline runs.")
    if args.dry_run:
        for n, L, seed, sub in runs:
            print(f"  n={n} L={L} seed={seed} -> results/{sub}/")
        return 0

    failed_runs: list[tuple[int, int, int, int]] = []
    skipped_runs: list[tuple[int, int, int]] = []
    for n, L, seed, sub in runs:
        target_dir = root / "results" / sub
        nlps_marker = target_dir / "nlps_final_scores.json"
        if args.skip_existing and not args.force and nlps_marker.is_file():
            print(
                f"\n=== SKIP: {sub} (nlps_final_scores.json already present) ===",
                flush=True,
            )
            skipped_runs.append((n, L, seed))
            continue

        env = os.environ.copy()
        env["QNLEP_SEED"] = str(seed)
        env["QNLEP_N_QUBITS"] = str(n)
        env["QNLEP_N_LAYERS"] = str(L)
        env["QNLEP_RESULTS_SUBDIR"] = sub
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(root / "scripts" / "run_neurips_pipeline.py"),
        ]
        inferred = _infer_start_from(target_dir) if not args.start_from else None
        if args.start_from:
            cmd.extend(["--start-from", args.start_from])
        elif inferred:
            cmd.extend(["--start-from", inferred])
        if args.through:
            cmd.extend(["--through", args.through])
        extra = []
        start_used = args.start_from or inferred
        if start_used:
            extra.append(f"start-from={start_used}")
        if args.through:
            extra.append(f"through={args.through}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        print(f"\n=== Running pipeline: {sub}{suffix} ===", flush=True)
        rc = subprocess.run(cmd, cwd=str(root), env=env)
        if rc.returncode != 0:
            print(f"FAILED: return code {rc.returncode} for {sub}", flush=True)
            failed_runs.append((n, L, seed, int(rc.returncode)))
            if not args.continue_on_error:
                return int(rc.returncode)

    print("\n--- Done. Suggested aggregation (from repo root) ---", flush=True)
    metrics = [
        "nlc_scores.json",
        "bppi_scores.json",
        "nss_scores.json",
        "ter_scores.json",
        "nlps_final_scores.json",
    ]
    for n, L in nl_pairs:
        g = f"results/sweeps/n{n}_L{L}/seed_*/"
        out_dir = f"results/sweeps/n{n}_L{L}"
        for m in metrics:
            stem = m.replace(".json", "")
            print(
                f'python scripts/aggregate_metric_jsons.py --glob "{g}{m}" '
                f'--out {out_dir}/{stem}_stats.json'
            )

    if skipped_runs:
        print("\nSkipped (already complete):", skipped_runs, flush=True)
    if failed_runs:
        print("\nFailed pipelines:", failed_runs, flush=True)

    finalize = os.environ.get("QNLEP_SWEEP_FINALIZE_PAPER", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if finalize:
        print("\n--- QNLEP_SWEEP_FINALIZE_PAPER: exporting LaTeX tables ---", flush=True)
        for n, L in nl_pairs:
            rc2 = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(root / "scripts" / "paper_export_multiseed_tables.py"),
                    "--n",
                    str(n),
                    "--l",
                    str(L),
                ],
                cwd=str(root),
            )
            if rc2.returncode != 0:
                return int(rc2.returncode)

    return 0 if not failed_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
