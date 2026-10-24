#!/usr/bin/env python3
"""Rebuild ``figures/real_data_heatmap.pdf`` from ``real_data_scores.json``.

Row order matches Table~\\ref{tab:real-data} / ``paper_export_multiseed_tables.py``.

Modes:

- Single seed: ``--json path/to/real_data_scores.json``
- Same numbers as Table~\\ref{tab:real-data} (mean over sweep seeds): ``--aggregate``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

# Same keys as scripts/paper_export_multiseed_tables.py REAL_ORDER_QUANTUM +
# CLASSICAL_METHODS (machine keys, not display labels).
METHOD_ROW_KEYS: tuple[str, ...] = (
    "strongly_entangling",
    "data_reuploading",
    "ancilla_tracing",
    "ancilla_reuploading",
    "qcnn",
    "quantum_reservoir",
    "classical_mlp",
    "hybrid_hqnn",
)

DATASET_ORDER: tuple[str, ...] = (
    "circles",
    "moons",
    "iris_1v2",
    "x_sin_x",
    "mackey_glass",
)

TASK_MAP: dict[str, str] = {
    "circles": "classification",
    "moons": "classification",
    "iris_1v2": "classification",
    "mackey_glass": "regression",
    "x_sin_x": "regression",
}


def _discover_seed_dirs(sweep_root: Path) -> list[Path]:
    if not sweep_root.is_dir():
        return []
    return sorted(
        p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )


def _parse_seed_int(name: str) -> int | None:
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def aggregate_real_data_scores(
    seed_dirs: list[Path],
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean test score per (dataset, method), same finite semantics as table export."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for ds in DATASET_ORDER:
        out[ds] = {}
        for mk in METHOD_ROW_KEYS:
            vals: list[float] = []
            for d in seed_dirs:
                path = d / "real_data_scores.json"
                if not path.is_file():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                block = data.get(ds, {}).get(mk)
                if isinstance(block, dict):
                    s = block.get("score")
                    if isinstance(s, (int, float)) and not isinstance(s, bool):
                        v = float(s)
                        if math.isfinite(v):
                            vals.append(v)
            if vals:
                out[ds][mk] = {"score": float(sum(vals) / len(vals))}
    return {k: v for k, v in out.items() if v}


def _cell_txt(v: float) -> str:
    if not math.isfinite(v) or math.isnan(v):
        return "—"
    return f"{v:.2f}"


def _score(data: dict, ds: str, m: str) -> float:
    block = data.get(ds, {}).get(m)
    if isinstance(block, dict) and isinstance(block.get("score"), (int, float)):
        return float(block["score"])
    return float("nan")


def plot_real_data_heatmap(data: dict, out_path: Path) -> int:
    dataset_names = [d for d in DATASET_ORDER if d in data and data[d]]
    cls_ds = [d for d in dataset_names if TASK_MAP[d] == "classification"]
    reg_ds = [d for d in dataset_names if TASK_MAP[d] == "regression"]
    reg_not_mg = [d for d in reg_ds if d != "mackey_glass"]
    has_mg = "mackey_glass" in reg_ds

    method_names = list(METHOD_ROW_KEYS)
    n_panels = int(bool(cls_ds)) + int(bool(reg_not_mg)) + int(has_mg)
    if n_panels == 0:
        print("No datasets to plot", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.2 * n_panels, len(method_names) * 0.72 + 1.3),
    )
    axes = np.atleast_1d(axes)
    ip = 0

    if cls_ds:
        m_cls = np.array(
            [[_score(data, ds, m) for ds in cls_ds] for m in method_names]
        )
        m_cls_plot = ma.masked_invalid(m_cls)
        im0 = axes[ip].imshow(m_cls_plot, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
        axes[ip].set_xticks(range(len(cls_ds)))
        axes[ip].set_xticklabels([d.replace("_", "\n") for d in cls_ds], fontsize=9)
        axes[ip].set_yticks(range(len(method_names)))
        axes[ip].set_yticklabels([m.replace("_", " ") for m in method_names], fontsize=9)
        for i in range(len(method_names)):
            for j in range(len(cls_ds)):
                axes[ip].text(
                    j,
                    i,
                    _cell_txt(m_cls[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        plt.colorbar(im0, ax=axes[ip], fraction=0.046, pad=0.04, label="Test accuracy")
        axes[ip].set_title("Classification", fontsize=10)
        ip += 1

    if reg_not_mg:
        m_x = np.array(
            [[_score(data, ds, m) for ds in reg_not_mg] for m in method_names]
        )
        finite = m_x[np.isfinite(m_x)]
        if finite.size:
            xa, xb = float(np.nanmin(m_x)), float(np.nanmax(m_x))
        else:
            xa, xb = 0.0, 1.0
        pad = 0.08 * (xb - xa + 1e-9)
        m_x_plot = ma.masked_invalid(m_x)
        im1 = axes[ip].imshow(
            m_x_plot, vmin=xa - pad, vmax=xb + pad, cmap="RdYlGn", aspect="auto"
        )
        axes[ip].set_xticks(range(len(reg_not_mg)))
        axes[ip].set_xticklabels([d.replace("_", "\n") for d in reg_not_mg], fontsize=9)
        axes[ip].set_yticks(range(len(method_names)))
        axes[ip].set_yticklabels([])
        for i in range(len(method_names)):
            for j in range(len(reg_not_mg)):
                axes[ip].text(
                    j,
                    i,
                    _cell_txt(m_x[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        plt.colorbar(im1, ax=axes[ip], fraction=0.046, pad=0.04, label=r"Test $R^2$")
        axes[ip].set_title(r"Regression ($x\sin x$)", fontsize=10)
        ip += 1

    if has_mg:
        vals = np.array(
            [_score(data, "mackey_glass", m) for m in method_names],
            dtype=float,
        )
        ypos = np.arange(len(method_names))
        plot_vals = np.where(np.isfinite(vals), vals, 0.0)
        axes[ip].barh(ypos, plot_vals, color="steelblue", edgecolor="k", linewidth=0.3)
        axes[ip].set_yticks(ypos)
        axes[ip].set_yticklabels([m.replace("_", " ") for m in method_names], fontsize=9)
        axes[ip].set_xlabel(r"Mackey--Glass test $R^2$ (not accuracy)", fontsize=9)
        axes[ip].set_title("Mackey--Glass (hard regression)", fontsize=10)
        axes[ip].axvline(0, color="k", lw=0.8, ls="--")
        axes[ip].grid(True, axis="x", alpha=0.3)
        for yi, v in enumerate(vals):
            if not np.isfinite(v):
                axes[ip].text(
                    0,
                    yi,
                    "—",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

    fig.suptitle(
        "Real-data scores: accuracy vs.\\ $R^2$ by task type",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Single-seed real_data_scores.json (ignored when --aggregate is set).",
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Average scores over seed_* dirs (matches multi-seed table export).",
    )
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--l", type=int, default=3)
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds to include (default: all seed_* under sweep root).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF (default: repo figures/real_data_heatmap.pdf).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_path = args.out or (root / "figures" / "real_data_heatmap.pdf")

    if args.aggregate:
        sweep_root = root / "results" / "sweeps" / f"n{args.n}_L{args.l}"
        dirs = _discover_seed_dirs(sweep_root)
        want = {int(x.strip()) for x in args.seeds.split(",") if x.strip()}
        if want:
            filt: list[Path] = []
            for p in dirs:
                sid = _parse_seed_int(p.name)
                if sid is not None and sid in want:
                    filt.append(p)
            dirs = filt
        if not dirs:
            print(f"No seed directories under {sweep_root}", file=sys.stderr)
            return 1
        data = aggregate_real_data_scores(dirs)
        print(
            f"aggregate seeds={[ _parse_seed_int(p.name) for p in dirs ]} "
            f"n_dirs={len(dirs)}",
            flush=True,
        )
    else:
        if args.json is None:
            print("Provide --json or --aggregate", file=sys.stderr)
            return 1
        data = json.loads(args.json.read_text(encoding="utf-8"))

    return plot_real_data_heatmap(data, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
