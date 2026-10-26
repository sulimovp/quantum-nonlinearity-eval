#!/usr/bin/env python3
"""
Build LaTeX table fragments from one or more protocol sweep directories.

Looks under ``results/sweeps/n{n}_L{L}/seed_*`` (default n=3,L=3). If no sweep
exists, falls back to ``results/`` (single-seed point estimates).

Writes:

- ``neurips2026/generated/metric_tables.meta.json`` — seed list, counts, UTC time
- ``neurips2026/generated/nlps_components_body.tex``
- ``neurips2026/generated/lorenz_r2_body.tex``
- ``neurips2026/generated/real_data_body.tex``
- ``neurips2026/generated/table_agg_blurb.tex``
- ``neurips2026/generated/table_agg_blurb.tex`` — short caption fragment for all numeric tables
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUANTUM_METHODS: tuple[tuple[str, str], ...] = (
    ("Strongly entangling (baseline)", "strongly_entangling"),
    ("Data reuploading", "data_reuploading"),
    ("Ancilla tracing", "ancilla_tracing"),
    ("Ancilla reuploading", "ancilla_reuploading"),
    ("Quantum CNN", "qcnn"),
    ("Quantum reservoir", "quantum_reservoir"),
)

CLASSICAL_METHODS: tuple[tuple[str, str], ...] = (
    ("Classical MLP (ref.)", "classical_mlp"),
    ("Hybrid QCNN (ref.)", "hybrid_hqnn"),
)

REAL_ORDER_QUANTUM: tuple[tuple[str, str], ...] = (
    ("Strongly entangling (baseline)", "strongly_entangling"),
    ("Data reuploading", "data_reuploading"),
    ("Ancilla tracing", "ancilla_tracing"),
    ("Ancilla reuploading", "ancilla_reuploading"),
    ("Quantum CNN", "qcnn"),
    ("Quantum reservoir", "quantum_reservoir"),
)

REAL_DATASETS: tuple[tuple[str, str], ...] = (
    ("circles", "Circles"),
    ("moons", "Moons"),
    ("iris_1v2", "Iris (1v2)"),
    ("x_sin_x", r"$x\sin(x)$"),
    ("mackey_glass", "Mackey--Glass"),
)


def _finite(xs: list[float]) -> list[float]:
    return [x for x in xs if not math.isnan(x)]


def _mean_std(xs: list[float]) -> tuple[float | None, float | None, int]:
    f = _finite(xs)
    n = len(f)
    if n == 0:
        return None, None, 0
    mu = sum(f) / n
    if n < 2:
        return mu, 0.0, n
    sig = statistics.stdev(f)
    return mu, sig, n


def _fmt_num(mu: float | None, sig: float | None, n: int, nd: int = 3) -> str:
    if mu is None:
        return "---"
    if n < 2 or sig is None:
        return f"{mu:.{nd}f}"
    return f"{mu:.{nd}f} $\\pm$ {sig:.{nd}f}"


def _fmt_intish(mu: float | None, sig: float | None, n: int) -> str:
    if mu is None:
        return "---"
    if n < 2 or sig is None:
        return str(int(round(mu)))
    return f"{mu:.0f} $\\pm$ {sig:.0f}"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _discover_seed_dirs(sweep_root: Path) -> list[Path]:
    if not sweep_root.is_dir():
        return []
    dirs = sorted(p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("seed_"))
    return dirs


def _parse_seed_csv(s: str | None) -> set[int] | None:
    if s is None or not str(s).strip():
        return None
    out: set[int] = set()
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        out.add(int(p))
    return out


def _filter_sweep_seeds(
    dirs: list[Path],
    include: set[int] | None,
    exclude: set[int],
) -> list[Path]:
    """
    Keep only ``seed_<int>`` paths. If *include* is set, require seed in the set;
    then drop any seed in *exclude* (e.g. legacy runs with a different real-data
    step budget so mean±std stays comparable).
    """
    if include is None and not exclude:
        return dirs
    out: list[Path] = []
    for p in dirs:
        if not p.name.startswith("seed_"):
            continue
        try:
            sid = int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if include is not None and sid not in include:
            continue
        if sid in exclude:
            continue
        out.append(p)
    return out


def _collect_lorenz_r2(seed_dirs: list[Path], method_key: str) -> list[float]:
    xs: list[float] = []
    for d in seed_dirs:
        path = d / "lorenz_results.json"
        if not path.is_file():
            continue
        data = _load_json(path)
        block = data.get(method_key)
        if not isinstance(block, dict):
            continue
        v = block.get("r2_test")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            xs.append(float(v))
        else:
            xs.append(math.nan)
    return xs


def _collect_real_dataset(seed_dirs: list[Path], dataset: str, method_key: str) -> list[float]:
    xs: list[float] = []
    for d in seed_dirs:
        path = d / "real_data_scores.json"
        if not path.is_file():
            continue
        data = _load_json(path)
        block = data.get(dataset)
        if not isinstance(block, dict):
            continue
        m = block.get(method_key)
        if isinstance(m, dict):
            v = m.get("score")
        else:
            v = None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            xs.append(float(v))
        else:
            xs.append(math.nan)
    return xs


def _build_protocol_rows(seed_dirs: list[Path]) -> list[str]:
    """Return tabular body lines (one per quantum method)."""
    # For each method, collect lists per metric
    metrics_cols: list[tuple[str, str, Any]] = [
        ("nlc", "nlc", lambda d, mk: _scalar_from_file(d, "nlc_scores.json", mk)),
        ("1mb", "1-bppi", lambda d, mk: _one_minus_bppi(d, mk)),
        ("ter", "ter", lambda d, mk: _scalar_from_file(d, "ter_scores.json", mk)),
        ("nss", "nss", lambda d, mk: _scalar_from_file(d, "nss_scores.json", mk)),
        ("nlps", "nlps", lambda d, mk: _nlps_field(d, mk, "NLPS")),
    ]

    grid: dict[str, dict[str, list[float]]] = {mk: {c[0]: [] for c in metrics_cols} for _, mk in QUANTUM_METHODS}

    for d in seed_dirs:
        for label, method_key in QUANTUM_METHODS:
            for col_key, _, fn in metrics_cols:
                v = fn(d, method_key)
                grid[method_key][col_key].append(v)

    col_best: dict[str, float | None] = {c[0]: None for c in metrics_cols}
    for col_key, _, _ in metrics_cols:
        best = None
        for _, mk in QUANTUM_METHODS:
            mu, _, n = _mean_std(grid[mk][col_key])
            if mu is None or n == 0:
                continue
            if best is None or mu > best:
                best = mu
        col_best[col_key] = best

    lines: list[str] = []
    for display, method_key in QUANTUM_METHODS:
        cells: list[str] = []
        for col_key, _, _ in metrics_cols:
            mu, sig, n = _mean_std(grid[method_key][col_key])
            s = _fmt_num(mu, sig, n, nd=3)
            bmean = col_best.get(col_key)
            if mu is not None and bmean is not None and abs(mu - bmean) < 1e-9:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append(f"    {display:22} & {' & '.join(cells)} \\\\")

    return lines


def _scalar_from_file(d: Path, fname: str, method_key: str) -> float:
    path = d / fname
    if not path.is_file():
        return math.nan
    data = _load_json(path)
    v = data.get(method_key)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return math.nan


def _one_minus_bppi(d: Path, method_key: str) -> float:
    v = _scalar_from_file(d, "bppi_scores.json", method_key)
    if math.isnan(v):
        return math.nan
    return 1.0 - v


def _nlps_field(d: Path, method_key: str, field: str) -> float:
    path = d / "nlps_final_scores.json"
    if not path.is_file():
        return math.nan
    data = _load_json(path)
    block = data.get(method_key)
    if not isinstance(block, dict):
        return math.nan
    v = block.get(field)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return math.nan


def _build_lorenz_rows(seed_dirs: list[Path]) -> list[str]:
    best_mu = None
    mus: dict[str, float | None] = {}
    for _, mk in QUANTUM_METHODS:
        mu, _, n = _mean_std(_collect_lorenz_r2(seed_dirs, mk))
        mus[mk] = mu
        if mu is not None and (best_mu is None or mu > best_mu):
            best_mu = mu

    lines: list[str] = []
    for display, mk in QUANTUM_METHODS:
        mu, sig, n = _mean_std(_collect_lorenz_r2(seed_dirs, mk))
        s = _fmt_num(mu, sig, n, nd=3)
        if mu is not None and best_mu is not None and abs(mu - best_mu) < 1e-9:
            s = f"\\textbf{{{s}}}"
        lines.append(f"    {display:22} & {s} \\\\")
    return lines


def _build_real_data_rows(seed_dirs: list[Path]) -> list[str]:
    """Quantum rows first, \\midrule, then classical reference rows.

    Bold = best mean per column among quantum rows only (classical MLP stays
    plain so it is clearly a non-competing reference).
    """
    col_best: dict[str, float | None] = {ds: None for ds, _ in REAL_DATASETS}
    for ds, _ in REAL_DATASETS:
        for _, mk in REAL_ORDER_QUANTUM:
            mu, _, n = _mean_std(_collect_real_dataset(seed_dirs, ds, mk))
            if mu is None:
                continue
            b = col_best[ds]
            if b is None or mu > b:
                col_best[ds] = mu

    def _row(display: str, mk: str, allow_bold: bool) -> str:
        cells: list[str] = []
        for ds, _ in REAL_DATASETS:
            mu, sig, n = _mean_std(_collect_real_dataset(seed_dirs, ds, mk))
            if ds == "mackey_glass":
                s = _fmt_intish(mu, sig, n)
            else:
                s = _fmt_num(mu, sig, n, nd=2)
            if allow_bold:
                b = col_best[ds]
                if mu is not None and b is not None and abs(mu - b) < 1e-9:
                    s = f"\\textbf{{{s}}}"
            cells.append(s)
        return f"    {display:24} & {' & '.join(cells)} \\\\"

    lines: list[str] = []
    for display, mk in REAL_ORDER_QUANTUM:
        lines.append(_row(display, mk, allow_bold=True))
    lines.append("    \\midrule")
    for display, mk in CLASSICAL_METHODS:
        lines.append(_row(display, mk, allow_bold=False))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--l", type=int, default=3)
    ap.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help="Override sweep root (default: results/sweeps/n{n}_L{l})",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated QNLEP_SEED list to include only (e.g. 43,44,45,46). Default: all seed_* under sweep root.",
    )
    ap.add_argument(
        "--exclude-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to omit (e.g. 42) when aggregating. Use if an old run used a different real-data step budget than the current protocol (QNLEP_REAL_DATA_N_STEPS=72).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    sweep_root = args.sweep_root
    if sweep_root is None:
        sweep_root = root / "results" / "sweeps" / f"n{args.n}_L{args.l}"

    include = _parse_seed_csv(args.seeds)
    exclude = _parse_seed_csv(args.exclude_seeds) or set()
    n_before: int | None = None
    using_filter = (include is not None) or bool(exclude)

    seed_dirs = _discover_seed_dirs(sweep_root)
    try:
        source = str(sweep_root.relative_to(root))
    except ValueError:
        source = str(sweep_root)

    if not seed_dirs:
        # Fallback: single canonical results dir (treat as one pseudo-seed)
        canonical = root / "results"
        if (canonical / "nlc_scores.json").is_file():
            if using_filter:
                raise SystemExit(
                    "Cannot use --seeds / --exclude-seeds with results/ fallback (no seed_* subdirs). "
                    "Set --sweep-root to a path under results/sweeps/."
                )
            seed_dirs = [canonical]
            source = "results/ (fallback, no sweep subdirs)"
        else:
            raise SystemExit("No seed_* directories under sweep root and no results/nlc_scores.json.")

    if using_filter:
        n_before = len(seed_dirs)
        seed_dirs = _filter_sweep_seeds(seed_dirs, include, exclude)
        if not seed_dirs:
            raise SystemExit(
                f"No seed directories left after filter "
                f"(include={include!r} exclude={sorted(exclude)}; had {n_before} before filter)."
            )
        print(
            f"Filtered sweep dirs: {n_before} -> {len(seed_dirs)} (include={include!r} exclude={sorted(exclude)})",
            flush=True,
        )

    seeds_parsed: list[int] = []
    for p in seed_dirs:
        name = p.name
        if name.startswith("seed_"):
            try:
                seeds_parsed.append(int(name.split("_", 1)[1]))
            except (IndexError, ValueError):
                seeds_parsed.append(-1)
        else:
            seeds_parsed.append(-1)

    out_dir = root / "neurips2026" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "n_seed_dirs": len(seed_dirs),
        "seeds": seeds_parsed,
        "paths": [str(p.relative_to(root)) for p in seed_dirs],
    }
    if using_filter:
        meta["filter"] = {
            "include_seeds": sorted(include) if include is not None else None,
            "exclude_seeds": sorted(exclude) if exclude else None,
        }
        if n_before is not None:
            meta["n_seed_dirs_before_filter"] = n_before
    (out_dir / "metric_tables.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    proto_lines = _build_protocol_rows(seed_dirs)
    (out_dir / "nlps_components_body.tex").write_text("\n".join(proto_lines) + "\n", encoding="utf-8")

    lorenz_lines = _build_lorenz_rows(seed_dirs)
    (out_dir / "lorenz_r2_body.tex").write_text("\n".join(lorenz_lines) + "\n", encoding="utf-8")

    real_lines = _build_real_data_rows(seed_dirs)
    (out_dir / "real_data_body.tex").write_text("\n".join(real_lines) + "\n", encoding="utf-8")

    k = len(seed_dirs)
    if k >= 2 and all(s >= 0 for s in seeds_parsed):
        suffix = (
            f"Mean $\\pm$ one standard deviation over $K={k}$ protocol runs "
            f"(\\texttt{{QNLEP\\_SEED}} $\\in \\{{{', '.join(str(s) for s in sorted(seeds_parsed))}\\}}$)."
        )
    else:
        suffix = (
            "Point estimates from the available protocol artifacts "
            f"({k} run(s); see \\texttt{{generated/metric\\_tables.meta.json}}). "
            "Re-run \\texttt{scripts/paper\\_export\\_multiseed\\_tables.py} after a multi-seed sweep for mean$\\pm$std."
        )
    (out_dir / "table_agg_blurb.tex").write_text(suffix + "\n", encoding="utf-8")

    print(f"Wrote LaTeX fragments under {out_dir.relative_to(root)} ({k} seed dirs, source={source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
