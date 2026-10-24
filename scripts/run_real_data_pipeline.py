#!/usr/bin/env python3
"""Real-data sweep with optional backup, append-only history, and Windows background.

1. Backs up prior ``real_data_scores.json`` / ``real_data_ter_scores.json`` and
   optional heatmap PDFs into timestamped ``_backups/`` folders (nothing deleted).
2. Runs ``scripts/run_real_data_all_seeds.py`` (append log).
3. Appends one-line entries to ``pipeline_runs/real_data_master.log``.

Examples::

  python scripts/run_real_data_pipeline.py
  python scripts/run_real_data_pipeline.py --background
  python scripts/run_real_data_pipeline.py --no-backup --seeds 42
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from backup_real_data_artifacts import backup_figures, backup_sweep_json


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _append_master_log(root: Path, line: str) -> None:
    p = root / "pipeline_runs" / "real_data_master.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{ts} {line}\n")


def _detached_popen(cmd: list[str], cwd: Path) -> subprocess.Popen:
    """Start *cmd* so it survives parent exit (Windows + Unix)."""
    stdin = subprocess.DEVNULL
    kwargs: dict = {
        "cwd": str(cwd),
        "stdin": stdin,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        kwargs["creationflags"] = creationflags
        kwargs["close_fds"] = False
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _run_body(
    root: Path,
    run_tag: str,
    log_path: Path,
    seeds_csv: str,
    n: int,
    l: int,
    fast: bool,
    steps: int | None,
    backup: bool,
    backup_figs: bool,
) -> int:
    seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip()]
    if backup:
        dest, n_copied = backup_sweep_json(root, n, l, seeds, run_tag)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(
                f"backup json tag={run_tag} -> {dest.relative_to(root)} "
                f"files_copied={n_copied}\n"
            )
        if backup_figs:
            fig_paths = backup_figures(root, run_tag)
            with open(log_path, "a", encoding="utf-8") as lf:
                for fp in fig_paths:
                    lf.write(f"backup figure -> {fp.relative_to(root)}\n")

    args = [
        sys.executable,
        "-u",
        "-X",
        "utf8",
        str(root / "scripts" / "run_real_data_all_seeds.py"),
        "--n",
        str(n),
        "--l",
        str(l),
        "--seeds",
        seeds_csv,
        "--log",
        str(log_path),
    ]
    if fast:
        args.append("--fast")
    if steps is not None:
        args.extend(["--steps", str(steps)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n>>> pipeline invoking: {' '.join(args)}\n")
        lf.flush()
        rc = subprocess.run(
            args,
            cwd=str(root),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--l", type=int, default=3)
    ap.add_argument("--seeds", type=str, default="43,44,45,46,42")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Append-only log for this run (default: pipeline_runs/real_data_sweep_<UTC>.log).",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="Backup tag folder name (default: same UTC stamp as log file name).",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip copying prior JSON/PDFs into _backups/ before the sweep.",
    )
    ap.add_argument(
        "--backup-figures",
        action="store_true",
        help="With backup on, also copy existing heatmap PDFs into _backups/.",
    )
    ap.add_argument(
        "--background",
        action="store_true",
        help="Spawn a detached worker and exit immediately (Windows-friendly).",
    )
    ap.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    root = _root()
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.log is not None:
        lp = Path(args.log)
        log_path = lp.resolve() if lp.is_absolute() else (root / lp).resolve()
    else:
        log_path = (root / "pipeline_runs" / f"real_data_sweep_{run_ts}.log").resolve()
    run_tag = (args.tag or "").strip() or run_ts
    backup = not args.no_backup
    backup_figs = backup and args.backup_figures

    forward = [
        sys.executable,
        "-u",
        str(root / "scripts" / "run_real_data_pipeline.py"),
        "--worker",
        "--n",
        str(args.n),
        "--l",
        str(args.l),
        "--seeds",
        args.seeds,
        "--log",
        str(log_path),
        "--tag",
        run_tag,
    ]
    if args.fast:
        forward.append("--fast")
    if args.steps is not None:
        forward.extend(["--steps", str(args.steps)])
    if args.no_backup:
        forward.append("--no-backup")
    if args.backup_figures:
        forward.append("--backup-figures")

    if args.background and not args.worker:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n{'=' * 72}\n"
                f"DETACHED dispatch utc={run_ts} log={log_path.name}\n"
                f"{'=' * 72}\n"
            )
        _append_master_log(
            root,
            f"DETACH log={log_path.relative_to(root)} tag={run_tag} seeds={args.seeds}",
        )
        proc = _detached_popen(forward, root)
        print(f"Detached worker pid={proc.pid}")
        print(f"Log file: {log_path.resolve()}")
        print(f"Master history: {(root / 'pipeline_runs' / 'real_data_master.log').resolve()}")
        return 0

    if args.worker:
        _append_master_log(
            root,
            f"RUN_START log={log_path.relative_to(root)} tag={run_tag} "
            f"n={args.n} L={args.l} seeds={args.seeds}",
        )
    else:
        _append_master_log(
            root,
            f"RUN_START foreground log={log_path.relative_to(root)} tag={run_tag}",
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n{'=' * 72}\n"
            f"run_real_data_pipeline start utc={run_ts}\n"
            f"n={args.n} L={args.l} seeds={args.seeds} backup={backup} "
            f"backup_figures={backup_figs}\n"
            f"{'=' * 72}\n"
        )

    try:
        rc = _run_body(
            root,
            run_tag,
            log_path,
            args.seeds,
            args.n,
            args.l,
            args.fast,
            args.steps,
            backup,
            backup_figs,
        )
    except Exception as exc:
        _append_master_log(root, f"RUN_FAIL log={log_path.relative_to(root)} error={exc!r}")
        raise

    _append_master_log(root, f"RUN_END log={log_path.relative_to(root)} exit={rc}")
    print(f"Log: {log_path.resolve()}")
    print(f"Exit code: {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
