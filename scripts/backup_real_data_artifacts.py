#!/usr/bin/env python3
"""Copy existing real-data JSON artifacts before a new sweep (no deletion).

Backs up, per seed:

- ``real_data_scores.json``
- ``real_data_ter_scores.json``

into ``results/sweeps/n{n}_L{L}/_backups/<tag>/seed_<seed>/`` so canonical paths
can be overwritten without losing the previous run.

Optional: backup heatmap PDFs into ``figures/_backups/`` and
``neurips2026/figures/_backups/``."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

REAL_DATA_FILES = ("real_data_scores.json", "real_data_ter_scores.json")
FIGURE_NAMES = ("real_data_heatmap.pdf",)


def default_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_sweep_json(
    root: Path,
    n: int,
    l: int,
    seeds: list[int],
    tag: str,
) -> tuple[Path, int]:
    """Copy real-data JSON files for listed seeds. Returns (backup_dir, n_copied)."""
    sweep = root / "results" / "sweeps" / f"n{n}_L{l}"
    dest_root = sweep / "_backups" / tag
    n_copied = 0
    for seed in seeds:
        sdir = sweep / f"seed_{seed}"
        for name in REAL_DATA_FILES:
            src = sdir / name
            if not src.is_file():
                continue
            d = dest_root / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, d / name)
            n_copied += 1
    return dest_root, n_copied


def backup_figures(root: Path, tag: str) -> list[Path]:
    """Copy heatmap PDFs if present. Returns list of written backup paths."""
    out: list[Path] = []
    pairs = [
        (root / "figures", root / "figures" / "_backups"),
        (root / "neurips2026" / "figures", root / "neurips2026" / "figures" / "_backups"),
    ]
    for src_dir, backup_dir in pairs:
        for name in FIGURE_NAMES:
            src = src_dir / name
            if not src.is_file():
                continue
            backup_dir.mkdir(parents=True, exist_ok=True)
            dst = backup_dir / f"{tag}_{name}"
            shutil.copy2(src, dst)
            out.append(dst)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--l", type=int, default=3)
    ap.add_argument(
        "--seeds",
        type=str,
        default="43,44,45,46,42",
        help="Comma-separated seeds to back up (only these directories).",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="Backup folder name under _backups/ (default: UTC timestamp).",
    )
    ap.add_argument(
        "--figures",
        action="store_true",
        help="Also backup real_data_heatmap.pdf under figures/ and neurips2026/figures/.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    tag = (args.tag or "").strip() or default_tag()

    dest, n_copied = backup_sweep_json(root, args.n, args.l, seeds, tag)
    print(f"backup_sweep_json tag={tag} dest={dest} files_copied={n_copied}")

    if args.figures:
        fig_bak = backup_figures(root, tag)
        for p in fig_bak:
            print(f"backup_figure {p.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
