#!/usr/bin/env python3
"""
Run the full NeurIPS / QNLEP notebook pipeline in one shot.

Order matches ``NEURIPS_NEXT_RUNS.md`` and ``NEURIPS_PROTOCOL.md``:
  03 (NLC) → 04 (BPPI) → 07 (NSS) → 08 (TER / Lorenz) → 11 (PQK baselines) → 09 (NLPS).

Notebooks assume the kernel cwd is ``notebooks/`` (they use ``../results`` and
``../figures``). This script always runs ``jupyter execute`` with cwd set to
that directory.

Environment:
  QNLEP_SEED — global seed for notebooks that respect it (default: 42).
  QNLEP_N_QUBITS / QNLEP_N_LAYERS — optional width/layer overrides for protocol
  notebooks (see ``src/qnlep_run_paths.py``).
  QNLEP_RESULTS_SUBDIR — optional path segment under ``results/`` for sweep isolation
  (e.g. ``sweeps/n3_L3/seed_42``); used by ``scripts/run_qnlep_sweep.py``.
  Other protocol env vars (e.g. QNLEP_NLC_IGNORE_PARTIAL) are passed through.

  Notebook **10** (real data) optional tuning (unset = defaults in the notebook):
    QNLEP_REAL_DATA_FAST — if ``1``/``true``, drops slow methods and datasets, caps
    steps / train rows, and skips Mackey-Glass panels when that dataset is omitted
    (CI / smoke). **Omit for paper-quality runs.**
    QNLEP_REAL_DATA_N_STEPS, QNLEP_REAL_DATA_NSAMPLES, QNLEP_REAL_DATA_MAX_TRAIN,
    QNLEP_REAL_DATA_METHODS (comma list), QNLEP_REAL_DATA_DATASETS (comma list),
    QNLEP_REAL_DATA_BOUNDARY_STEPS — override training / plotting budgets.
    QNLEP_REAL_DATA_MG_TRAIN / QNLEP_REAL_DATA_MG_TEST — Mackey–Glass window sizes (notebook 10).

  On Windows, ``PYTHONUTF8=1``, ``PYTHONIOENCODING=utf-8``, and ``python -X utf8``
  are set for ``jupyter execute`` so ``.ipynb`` files are always read as UTF-8
  (avoids ``UnicodeDecodeError`` from the ANSI code page).

Logs:
  ``pipeline_runs/<UTC-timestamp>/pipeline.log`` — wrapper lines plus **captured**
  ``jupyter execute`` stdout/stderr (so detached / hidden-window runs still retain
  notebook print output and tracebacks).

Artifacts:
  JSON under ``results/``, PDFs under ``figures/`` (written by the notebooks),
  plus ``results/pipeline_run_summary.json`` when the run finishes (success or
  partial failure).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Order is protocol-critical (08 before 09 for TER in NLPS).
# 06 = hybrid HQNN + classical MLP baselines; 10 = real datasets (writes
# real_data_ter_scores.json — never overwrites protocol ter_scores.json from 08).
PIPELINE_NOTEBOOKS: tuple[str, ...] = (
    "03_nlc_fourier_capacity.ipynb",
    "04_bppi_barren_plateau.ipynb",
    "07_nss_noise_stability.ipynb",
    "08_lorenz_attractor.ipynb",
    "11_comparison_metrics.ipynb",
    "09_nlps_composite_score.ipynb",
    "06_hybrid_hqnn.ipynb",
    "10_real_data.ipynb",
)

# JSON files the paper workflow expects after a full successful run.
EXPECTED_RESULTS_JSON: tuple[str, ...] = (
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _notebooks_from_start(start_from: str | None) -> tuple[str, ...]:
    if not start_from:
        return PIPELINE_NOTEBOOKS
    if start_from not in PIPELINE_NOTEBOOKS:
        choices = ", ".join(PIPELINE_NOTEBOOKS)
        raise ValueError(f"Unknown notebook {start_from!r}. Choose one of: {choices}")
    idx = PIPELINE_NOTEBOOKS.index(start_from)
    return PIPELINE_NOTEBOOKS[idx:]


def _notebooks_through(through: str | None, notebooks: tuple[str, ...]) -> tuple[str, ...]:
    if through is None:
        return notebooks
    if through not in PIPELINE_NOTEBOOKS:
        choices = ", ".join(PIPELINE_NOTEBOOKS)
        raise ValueError(f"Unknown notebook {through!r}. Choose one of: {choices}")
    end = PIPELINE_NOTEBOOKS.index(through)
    start = PIPELINE_NOTEBOOKS.index(notebooks[0])
    if end < start:
        raise ValueError(f"--through {through!r} precedes pipeline start {notebooks[0]!r}")
    return tuple(n for n in notebooks if PIPELINE_NOTEBOOKS.index(n) <= end)


def _tee_print(msg: str, log_fp) -> None:
    line = msg.rstrip("\n")
    print(line, flush=True)
    if log_fp:
        log_fp.write(line + "\n")
        log_fp.flush()


def _run_notebook(
    nb_path: Path,
    cwd: Path,
    env: dict[str, str],
    log_fp,
    startup_timeout: int,
    mirror_console: bool,
) -> int:
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "jupyter",
        "execute",
        str(nb_path.name),
        "--timeout=-1",
        f"--startup_timeout={startup_timeout}",
    ]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _tee_print(f"\n>>> [{ts}] {' '.join(cmd)}", log_fp)
    _tee_print(f"    cwd={cwd}", log_fp)
    t0 = time.perf_counter()

    if not mirror_console:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        dt = time.perf_counter() - t0
        _tee_print(
            f"<<< [{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
            f"exit {p.returncode} after {dt:.1f}s",
            log_fp,
        )
        return int(p.returncode)

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    def _pump_stdout():
        assert p.stdout is not None
        for line in iter(p.stdout.readline, ""):
            log_fp.write(line)
            log_fp.flush()
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except BrokenPipeError:
                pass

    pump = threading.Thread(target=_pump_stdout, daemon=True)
    pump.start()
    rc = int(p.wait())
    pump.join(timeout=5.0)
    dt = time.perf_counter() - t0
    _tee_print(
        f"<<< [{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
        f"exit {rc} after {dt:.1f}s",
        log_fp,
    )
    return rc


def _artifact_report(results_dir: Path) -> dict:
    out: dict = {"results": {}, "missing_required": []}
    for name in EXPECTED_RESULTS_JSON:
        p = results_dir / name
        if p.is_file():
            st = p.stat()
            out["results"][name] = {
                "exists": True,
                "size_bytes": st.st_size,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        else:
            out["results"][name] = {"exists": False}
            out["missing_required"].append(name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("QNLEP_SEED", "42")),
        help="Sets QNLEP_SEED for subprocesses (default: env QNLEP_SEED or 42).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=300,
        help="Jupyter kernel startup timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned steps and exit without executing.",
    )
    parser.add_argument(
        "--no-console-mirror",
        action="store_true",
        help="Never copy jupyter stdout to the parent process (only pipeline.log). "
        "By default mirroring is on only when stdout is a TTY (so detached runs "
        "still get a full log file).",
    )
    parser.add_argument(
        "--start-from",
        metavar="NOTEBOOK.ipynb",
        default=None,
        help="Resume protocol from this notebook (must match a name in the pipeline "
        "tuple), e.g. 08_lorenz_attractor.ipynb after an earlier failure.",
    )
    parser.add_argument(
        "--through",
        metavar="NOTEBOOK.ipynb",
        default=None,
        help="Stop after this notebook (inclusive), e.g. 09_nlps_composite_score.ipynb "
        "to skip hybrid/real-data panels when only NLPS ranks are needed.",
    )
    args = parser.parse_args()

    try:
        notebooks_to_run = _notebooks_from_start(args.start_from)
        notebooks_to_run = _notebooks_through(args.through, notebooks_to_run)
    except ValueError as e:
        parser.error(str(e))

    root = _repo_root()
    nb_dir = root / "notebooks"
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "pipeline_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    (root / "pipeline_runs" / "LAST_RUN.txt").write_text(
        f"{log_path.resolve()}\n", encoding="utf-8"
    )

    summary = {
        "run_id": run_id,
        "repo_root": str(root),
        "notebook_dir": str(nb_dir),
        "qnlep_seed": args.seed,
        "start_from": args.start_from,
        "through": args.through,
        "notebooks_planned": list(notebooks_to_run),
        "steps": [],
        "final_status": "not_started",
    }

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["QNLEP_SEED"] = str(args.seed)
    # So imports work even if a notebook forgets sys.path.insert
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    with open(log_path, "w", encoding="utf-8") as log_fp:
        _tee_print(f"QNLEP NeurIPS pipeline run_id={run_id}", log_fp)
        _tee_print(f"Log file: {log_path}", log_fp)
        _tee_print(f"QNLEP_SEED={args.seed}", log_fp)
        if args.start_from:
            _tee_print(f"--start-from {args.start_from} (subset of {len(notebooks_to_run)} notebooks)", log_fp)
        if args.through:
            _tee_print(f"--through {args.through} (inclusive stop)", log_fp)

        if args.dry_run:
            for name in notebooks_to_run:
                _tee_print(f"  [dry-run] would execute: {name}", log_fp)
            _tee_print("Dry run complete (no notebooks executed; results/ unchanged).", log_fp)
            return 0

        overall_ok = True
        for name in notebooks_to_run:
            nb_path = nb_dir / name
            if not nb_path.is_file():
                _tee_print(f"ERROR: missing notebook {nb_path}", log_fp)
                summary["steps"].append({"notebook": name, "error": "file_not_found"})
                overall_ok = False
                break
            step_t0 = time.perf_counter()
            mirror = (not args.no_console_mirror) and sys.stdout.isatty()
            rc = _run_notebook(
                nb_path, nb_dir, env, log_fp, args.startup_timeout, mirror
            )
            elapsed = time.perf_counter() - step_t0
            summary["steps"].append(
                {"notebook": name, "returncode": rc, "elapsed_s": round(elapsed, 3)}
            )
            if rc != 0:
                overall_ok = False
                summary["final_status"] = f"failed_at_{name}"
                _tee_print(f"Pipeline stopped after failure in {name}", log_fp)
                break

        if overall_ok:
            summary["final_status"] = "success"

        summary["artifacts"] = _artifact_report(results_dir)
        summary["figures_pdf_count"] = len(list(figures_dir.glob("*.pdf")))

        out_summary_path = results_dir / "pipeline_run_summary.json"
        with open(out_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        _tee_print(f"Wrote {out_summary_path}", log_fp)

        if summary["artifacts"]["missing_required"]:
            _tee_print(
                "WARNING: missing JSON artifacts: "
                + ", ".join(summary["artifacts"]["missing_required"]),
                log_fp,
            )

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
