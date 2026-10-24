#!/usr/bin/env python3
"""BPPI probe-loss ablation: re-run gradient-variance sweep under alternate probes.

Writes ``results/bppi_probe_ablation/summary.json`` and
``neurips2026/generated/bppi_probe_ablation_table.tex``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy import stats
from scipy.stats import linregress

from src.metrics.protocol_scores import smooth_bppi_from_slope
from src.qnlep_registry import QNLEP_BUILDERS, param_shape

METHODS: tuple[str, ...] = (
    "data_reuploading",
    "ancilla_tracing",
    "ancilla_reuploading",
    "strongly_entangling",
    "qcnn",
    "quantum_reservoir",
)

N_QUBIT_SWEEP = [2, 4, 6, 8]
N_LAYERS = 3
N_RANDOM_INIT = 96  # ablation budget; default probe uses stored sweep slopes when available


def _probe_loss_fn(probe_id: str) -> Callable[[Callable, float], Callable]:
    if probe_id == "default":
        def loss_fn(circuit, x):
            return lambda w: (circuit(w, x) - 0.0) ** 2
        return loss_fn

    if probe_id.startswith("alt_x_"):
        x_val = float(probe_id.split("_", 2)[-1].replace("neg", "-"))
        def loss_fn(circuit, _x):
            return lambda w: (circuit(w, x_val) - 0.0) ** 2
        return loss_fn

    if probe_id == "local_mse":
        xs = np.linspace(-np.pi, np.pi, 5)

        def loss_fn(circuit, _x):
            def loss(w):
                errs = [(circuit(w, float(xi)) - float(np.sin(xi))) ** 2 for xi in xs]
                return sum(errs) / len(errs)
            return loss
        return loss_fn

    if probe_id == "observable_grad":
        def loss_fn(circuit, x):
            return lambda w: circuit(w, x)
        return loss_fn

    raise ValueError(f"Unknown probe_id {probe_id!r}")


def gradient_variance(
    builder,
    method_name: str,
    n_qubits: int,
    n_layers: int,
    probe_id: str,
    seed: int,
    n_init: int = N_RANDOM_INIT,
    x_probe: float = 0.5,
) -> float:
    rng = np.random.default_rng(seed + n_qubits * 1000)
    if method_name == "quantum_reservoir":
        circuit = builder(n_qubits, n_layers, reservoir_seed=seed + n_qubits)
    else:
        circuit = builder(n_qubits, n_layers)
    p_shape = param_shape(method_name, n_qubits, n_layers)
    loss_factory = _probe_loss_fn(probe_id)
    loss = loss_factory(circuit, x_probe)
    grad_fn = qml.grad(loss, argnum=0)
    grads = []
    for _ in range(n_init):
        w = pnp.array(rng.uniform(0, 2 * np.pi, p_shape), requires_grad=True)
        g = grad_fn(w)
        grads.append(float(g.ravel()[0]))
    return float(np.var(grads))


def compute_bppi(variances: list[float], n_values: list[int]) -> tuple[float, float]:
    log_v = [math.log2(max(v, 1e-30)) for v in variances]
    slope, intercept, r_value, _, _ = linregress(n_values, log_v)
    bppi = smooth_bppi_from_slope(slope)
    return float(bppi), float(slope)


def rank_by_one_minus_bppi(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda m: (1.0 - scores[m]), reverse=True)


def kendall_vs_default(rank_a: list[str], rank_b: list[str]) -> float:
    pos = {m: i for i, m in enumerate(rank_a)}
    a = [pos[m] for m in rank_b]
    tau, _ = stats.kendalltau(range(len(a)), a)
    return float(tau) if tau is not None and math.isfinite(tau) else math.nan


def _load_default_slopes_from_sweep(root: Path, seed: int) -> dict[str, float] | None:
    path = root / "results" / "sweeps" / "n3_L3" / f"seed_{seed}" / "bppi_detail.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    by_method = data.get("by_method", data)
    out: dict[str, float] = {}
    for m in METHODS:
        block = by_method.get(m, {})
        if isinstance(block, dict):
            slope = block.get("slope", block.get("slope_alpha"))
            if slope is not None:
                out[m] = float(slope)
    return out if out else None


def run_probe(
    probe_id: str,
    seeds: list[int],
    root: Path,
    n_init: int,
    n_qubit_sweep: list[int],
    *,
    recompute_default: bool = False,
) -> dict:
    """Estimate BPPI under ``probe_id``.

    For ``probe_id == "default"``, slopes are loaded from the published sweep
    unless ``recompute_default`` is set (the ablation-budget control).
    """
    per_seed: dict[str, dict[str, float]] = {}
    source = "recomputed"
    for seed in seeds:
        if probe_id == "default" and not recompute_default:
            slopes = _load_default_slopes_from_sweep(root, seed)
            if slopes:
                bppi_scores = {m: smooth_bppi_from_slope(slopes[m]) for m in METHODS}
                per_seed[str(seed)] = bppi_scores
                source = "published_sweep"
                continue
        bppi_scores: dict[str, float] = {}
        for method in METHODS:
            builder = QNLEP_BUILDERS[method]
            variances = [
                gradient_variance(
                    builder, method, n, N_LAYERS, probe_id, seed, n_init=n_init
                )
                for n in n_qubit_sweep
            ]
            bppi, _ = compute_bppi(variances, n_qubit_sweep)
            bppi_scores[method] = bppi
        per_seed[str(seed)] = bppi_scores

    # mean BPPI per method (ranking uses 1-BPPI)
    mean_scores: dict[str, float] = {}
    for m in METHODS:
        vals = [per_seed[str(s)][m] for s in seeds]
        mean_scores[m] = float(np.mean(vals))
    ranking = rank_by_one_minus_bppi(mean_scores)
    return {
        "per_seed": per_seed,
        "mean_bppi": mean_scores,
        "ranking": ranking,
        "source": source,
    }


def _write_latex(summary: dict, out: Path) -> None:
    probes = summary["probes"]
    ref_key = (
        "default_published"
        if "default_published" in probes
        else "default"
    )
    n_init = summary.get("n_init", "?")
    n_sweep = summary.get("n_qubit_sweep", [])
    seeds = summary.get("seeds", [])
    budget = (
        f"$N_{{\\mathrm{{init}}}}={n_init}$, "
        f"$n\\in\\{{{','.join(str(n) for n in n_sweep)}\\}}$, "
        f"seeds {', '.join(str(s) for s in seeds)}"
    )
    lines = [
        "% Auto-generated by scripts/bppi_probe_ablation.py",
        "\\begin{table}[t]",
        "  \\caption{BPPI rank bands under alternate probe losses at "
        "$(n,L)=(3,3)$. Ablation budget: "
        f"{budget}. "
        "Ranks follow $1-\\mathrm{BPPI}$ (higher is better). "
        "Kendall $\\tau$ is against the published default probe "
        "(full stage: $N_{\\mathrm{init}}=256$, $n\\in\\{2,4,6,8\\}$) "
        "on the same seeds. "
        "The row \\texttt{default (ablation budget)} re-runs the default "
        "probe at the ablation budget as a control.}",
        "  \\label{tab:bppi-probe-ablation}",
        "  \\centering",
        "  \\footnotesize",
        "  \\begin{tabular}{lrrl}",
        "    \\toprule",
        "    Probe & Kendall $\\tau$ vs published default "
        "& Top method & Bottom two (ranks 5--6) \\\\",
        "    \\midrule",
    ]
    row_order = [
        "default_published",
        "default_ablation_budget",
        "default",
        "alt_x_0.1",
        "alt_x_1.0",
        "local_mse",
    ]
    seen: set[str] = set()
    ordered = [p for p in row_order if p in probes] + [
        p for p in probes if p not in row_order
    ]
    for pid in ordered:
        if pid in seen:
            continue
        seen.add(pid)
        block = probes[pid]
        tau = block.get("kendall_tau_vs_default", float("nan"))
        rank = block["ranking"]
        top = rank[0].replace("_", " ")
        bottom = ", ".join(r.replace("_", " ") for r in rank[-2:])
        tau_s = f"{tau:.2f}" if math.isfinite(tau) else "---"
        label = {
            "default_published": "default (published)",
            "default_ablation_budget": "default (ablation budget)",
            "default": "default",
            "alt_x_0.1": "alt $x{=}0.1$",
            "alt_x_1.0": "alt $x{=}1.0$",
            "local_mse": "local $\\sin$ MSE",
        }.get(pid, pid.replace("_", " "))
        lines.append(f"    {label} & {tau_s} & {top} & {bottom} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--n-init", type=int, default=16)
    ap.add_argument(
        "--probes",
        default="default,alt_x_0.1,alt_x_1.0,local_mse",
    )
    ap.add_argument(
        "--n-qubits",
        default="2,4,6",
        help="Comma-separated qubit counts for the BPPI sweep.",
    )
    ap.add_argument(
        "--recompute-default",
        action="store_true",
        help="Re-estimate the default probe at the ablation budget "
        "(do not load published sweep slopes).",
    )
    ap.add_argument(
        "--with-published-default",
        action="store_true",
        help="Also load the published default as a separate reference row.",
    )
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    n_qubit_sweep = [int(x.strip()) for x in args.n_qubits.split(",") if x.strip()]

    root = _ROOT
    os.chdir(root)

    results: dict[str, dict] = {}
    if args.with_published_default or args.recompute_default:
        print("Loading published default probe ...", flush=True)
        results["default_published"] = run_probe(
            "default", seeds, root, args.n_init, n_qubit_sweep,
            recompute_default=False,
        )

    for pid in probes:
        force = args.recompute_default and pid == "default"
        label = (
            "default_ablation_budget"
            if force and "default_published" in results
            else pid
        )
        print(
            f"Running probe {pid}"
            f"{' (recompute at ablation budget)' if force else ''} ...",
            flush=True,
        )
        block = run_probe(
            pid, seeds, root, args.n_init, n_qubit_sweep,
            recompute_default=force,
        )
        results[label] = block

    ref_key = (
        "default_published"
        if "default_published" in results
        else ("default" if "default" in results else next(iter(results)))
    )
    default_rank = results[ref_key]["ranking"]
    for pid, block in results.items():
        block["kendall_tau_vs_default"] = kendall_vs_default(
            default_rank, block["ranking"]
        )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "n_init": args.n_init,
        "n_qubit_sweep": n_qubit_sweep,
        "n_layers": N_LAYERS,
        "reference_ranking": ref_key,
        "probes": results,
    }
    out_dir = root / "results" / "bppi_probe_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tex = root / "neurips2026" / "generated" / "bppi_probe_ablation_table.tex"
    _write_latex(summary, tex)
    print(f"Wrote {out_dir / 'summary.json'} and {tex}")
    # print control comparison for the reply
    if "default_ablation_budget" in results and "default_published" in results:
        tau = results["default_ablation_budget"]["kendall_tau_vs_default"]
        print(
            "CONTROL: cheap default vs published default Kendall tau = "
            f"{tau:.3f}"
        )
        print("  published rank:", results["default_published"]["ranking"])
        print("  cheap rank:    ", results["default_ablation_budget"]["ranking"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
