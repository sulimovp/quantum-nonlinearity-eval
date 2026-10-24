#!/usr/bin/env python3
"""Run ``scripts/real_data_headless.py`` per sweep seed (matches notebook 10 numerics).

Writes ``real_data_scores.json`` under ``results/sweeps/n{n}_L{L}/seed_<seed>/``,
runs ``paper_export_multiseed_tables.py``, then rebuilds ``figures/real_data_heatmap.pdf``
by **averaging the same seeds** (so the heatmap matches Table~\\ref{tab:real-data}),
and copies the heatmap into ``neurips2026/figures/``.

Example::

  python scripts/run_real_data_all_seeds.py --log pipeline_runs/real_data_10.log
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n",
        type=int,
        default=3,
        help="QNLEP_N_QUBITS / protocol width (default: 3).",
    )
    ap.add_argument(
        "--l",
        type=int,
        default=3,
        help="QNLEP_N_LAYERS (default: 3).",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="43,44,45,46,42",
        help=(
            "Comma-separated seeds; default ends with 42 (heatmap source for PDF)."
        ),
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Set QNLEP_REAL_DATA_FAST=1 (smaller budget; not for final paper).",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Set QNLEP_REAL_DATA_N_STEPS (default: unset → notebook default 72).",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Append full notebook stdout/stderr here (default: pipeline_runs/ts log).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.log or (root / "pipeline_runs" / f"real_data_nb10_{run_ts}.log")

    env_base = os.environ.copy()
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base["PYTHONUTF8"] = "1"
    env_base["PYTHONIOENCODING"] = "utf-8"
    env_base["PYTHONPATH"] = str(root) + os.pathsep + env_base.get("PYTHONPATH", "")
    env_base["QNLEP_N_QUBITS"] = str(args.n)
    env_base["QNLEP_N_LAYERS"] = str(args.l)
    if args.fast:
        env_base["QNLEP_REAL_DATA_FAST"] = "1"
    else:
        env_base.pop("QNLEP_REAL_DATA_FAST", None)
    if args.steps is not None:
        env_base["QNLEP_REAL_DATA_N_STEPS"] = str(args.steps)
    else:
        env_base.pop("QNLEP_REAL_DATA_N_STEPS", None)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fails: list[tuple[int, int]] = []

    with open(log_path, "a", encoding="utf-8") as log_fp:
        log_fp.write(
            f"\n{'=' * 72}\n"
            f"run_real_data_all_seeds start utc={run_ts}\n"
            f"n={args.n} L={args.l} seeds={seeds} fast={args.fast} steps={args.steps}\n"
            f"{'=' * 72}\n\n"
        )
        log_fp.flush()

        headless = root / "scripts" / "real_data_headless.py"
        for seed in seeds:
            env = env_base.copy()
            env["QNLEP_SEED"] = str(seed)
            env["QNLEP_RESULTS_SUBDIR"] = f"sweeps/n{args.n}_L{args.l}/seed_{seed}"
            cmd = [
                sys.executable,
                "-u",
                "-X",
                "utf8",
                str(headless),
                "--seed",
                str(seed),
            ]
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            log_fp.write(f"\n>>> [{ts}] seed={seed} QNLEP_RESULTS_SUBDIR={env['QNLEP_RESULTS_SUBDIR']}\n")
            log_fp.write(f"    {' '.join(cmd)}\n")
            log_fp.write(f"    cwd={root}\n")
            log_fp.flush()

            t0 = datetime.now(timezone.utc)
            p = subprocess.run(
                cmd,
                cwd=str(root),
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
            dt = (datetime.now(timezone.utc) - t0).total_seconds()
            log_fp.write(
                f"<<< exit {p.returncode} after {dt:.1f}s\n",
            )
            log_fp.flush()
            if p.returncode != 0:
                fails.append((seed, p.returncode))

        if not fails:
            export_cmd = [
                sys.executable,
                "-u",
                "-X",
                "utf8",
                str(root / "scripts" / "paper_export_multiseed_tables.py"),
                "--n",
                str(args.n),
                "--l",
                str(args.l),
            ]
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            log_fp.write(f"\n>>> [{ts}] paper_export_multiseed_tables (before heatmap)\n")
            log_fp.write(f"    {' '.join(export_cmd)}\n")
            log_fp.flush()
            p_ex = subprocess.run(
                export_cmd,
                cwd=str(root),
                env=env_base,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_fp.write(f"<<< paper_export exit {p_ex.returncode}\n")
            if p_ex.returncode != 0:
                fails.append((-1, p_ex.returncode))
            else:
                seeds_csv = ",".join(str(s) for s in seeds)
                plot_cmd = [
                    sys.executable,
                    "-u",
                    "-X",
                    "utf8",
                    str(root / "scripts" / "plot_real_data_heatmap.py"),
                    "--aggregate",
                    "--n",
                    str(args.n),
                    "--l",
                    str(args.l),
                    "--seeds",
                    seeds_csv,
                    "--out",
                    str(root / "figures" / "real_data_heatmap.pdf"),
                ]
                ts2 = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                log_fp.write(
                    f"\n>>> [{ts2}] heatmap (mean over seeds, matches table)\n"
                )
                log_fp.write(f"    {' '.join(plot_cmd)}\n")
                log_fp.flush()
                p2 = subprocess.run(
                    plot_cmd,
                    cwd=str(root),
                    env=env_base,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                log_fp.write(f"<<< plot exit {p2.returncode}\n")
                if p2.returncode != 0:
                    fails.append((-2, p2.returncode))

        log_fp.write(f"\nDone. Failures: {fails}\n")

    print(f"Log: {log_path.resolve()}")
    if fails:
        print(f"FAILED seeds: {fails}", file=sys.stderr)
        return 1

    src_hm = root / "figures" / "real_data_heatmap.pdf"
    dst_hm = root / "neurips2026" / "figures" / "real_data_heatmap.pdf"
    if src_hm.is_file():
        dst_hm.parent.mkdir(parents=True, exist_ok=True)
        dst_hm.write_bytes(src_hm.read_bytes())
        print(f"Copied {src_hm.relative_to(root)} -> {dst_hm.relative_to(root)}")
    else:
        print(f"WARNING: missing {src_hm}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
