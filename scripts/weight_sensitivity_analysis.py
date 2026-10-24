#!/usr/bin/env python3
"""Run NLPS and TER weight-sensitivity analyses from saved JSON artifacts.

The script does not train circuits. It reuses the scalar protocol outputs in
``results/`` and recomputes rankings under admissible weight perturbations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import linprog


NLPS_COMPONENTS = ("NLC", "1-BPPI", "TER", "NSS")
NLPS_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.20], dtype=float)
TER_COMPONENTS = ("curve_score", "best_train_score", "test_score")
TER_WEIGHTS = np.array([0.50, 0.30, 0.20], dtype=float)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _as_float(value, field: str, method: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        x = float(value)
        if math.isfinite(x):
            return x
    raise ValueError(f"Missing finite {field!r} for method {method!r}")


def _load_nlps_components(results_dir: Path) -> tuple[list[str], np.ndarray]:
    data = _load_json(results_dir / "nlps_final_scores.json")
    methods = [m for m, block in data.items() if isinstance(block, dict)]
    if not methods:
        raise ValueError("No method blocks found in nlps_final_scores.json")

    rows: list[list[float]] = []
    for method in methods:
        block = data[method]
        rows.append(
            [
                _as_float(block.get(component), component, method)
                for component in NLPS_COMPONENTS
            ]
        )
    return methods, np.array(rows, dtype=float)


def _load_ter_components(
    results_dir: Path,
    methods: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    data = _load_json(results_dir / "lorenz_results.json")
    rows: list[list[float]] = []
    multipliers: list[float] = []
    for method in methods:
        block = data.get(method)
        if not isinstance(block, dict):
            raise ValueError(f"Missing Lorenz block for method {method!r}")
        rows.append(
            [
                _as_float(block.get(component), component, method)
                for component in TER_COMPONENTS
            ]
        )
        multipliers.append(
            _as_float(block.get("param_efficiency"), "param_efficiency", method)
        )
    return np.array(rows, dtype=float), np.array(multipliers, dtype=float)


def _sample_until(
    n_samples: int,
    make_batch: Callable[[int], np.ndarray],
    predicate: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
) -> np.ndarray:
    accepted: list[np.ndarray] = []
    n_accepted = 0
    while n_accepted < n_samples:
        batch = make_batch(batch_size)
        mask = predicate(batch)
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        if np.any(mask):
            chosen = batch[mask]
            accepted.append(chosen)
            n_accepted += chosen.shape[0]

    return np.vstack(accepted)[:n_samples]


def _sample_global_nlps_weights(
    rng: np.random.Generator,
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    min_weight = 0.15

    def make_batch(size: int) -> np.ndarray:
        residual = 1.0 - 4.0 * min_weight
        return min_weight + residual * rng.dirichlet(np.ones(4), size=size)

    def predicate(w: np.ndarray) -> np.ndarray:
        return (
            (w[:, 0] > w[:, 1])
            & (w[:, 1] >= w[:, 2])
            & (w[:, 1] >= w[:, 3])
        )

    return _sample_until(n_samples, make_batch, predicate, batch_size)


def _sample_local_nlps_weights(
    rng: np.random.Generator,
    n_samples: int,
    epsilon: float,
    batch_size: int,
) -> np.ndarray:
    base = NLPS_WEIGHTS

    def make_batch(size: int) -> np.ndarray:
        delta = rng.uniform(-epsilon, epsilon, size=(size, 3))
        last = -np.sum(delta, axis=1, keepdims=True)
        return base + np.hstack([delta, last])

    def predicate(w: np.ndarray) -> np.ndarray:
        return (
            (np.abs(w - base) <= epsilon + 1e-12).all(axis=1)
            & (w >= 0.0).all(axis=1)
            & np.isclose(np.sum(w, axis=1), 1.0)
            & (w[:, 0] > w[:, 1])
            & (w[:, 1] >= w[:, 2])
            & (w[:, 1] >= w[:, 3])
        )

    return _sample_until(n_samples, make_batch, predicate, batch_size)


def _sample_ter_weights(
    rng: np.random.Generator,
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    def make_batch(size: int) -> np.ndarray:
        return rng.dirichlet(np.ones(3), size=size)

    def predicate(w: np.ndarray) -> np.ndarray:
        return (
            (w[:, 0] > w[:, 1])
            & (w[:, 1] > w[:, 2])
            & (w[:, 0] >= 0.40)
            & (w[:, 2] >= 0.10)
        )

    return _sample_until(n_samples, make_batch, predicate, batch_size)


def _base_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _rank_matrix(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    row_idx = np.arange(scores.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, scores.shape[1] + 1)
    return ranks


def _rank_summary(
    methods: list[str],
    scores: np.ndarray,
    reference_scores: np.ndarray,
) -> dict:
    ranks = _rank_matrix(scores)
    n_samples, n_methods = scores.shape
    reference_ranks = _base_ranks(reference_scores)

    rank_frequency: dict[str, dict[str, float | int]] = {}
    for idx, method in enumerate(methods):
        counts = np.bincount(ranks[:, idx], minlength=n_methods + 1)
        block: dict[str, float | int] = {
            f"rank_{rank}": float(counts[rank] / n_samples)
            for rank in range(1, n_methods + 1)
        }
        block["modal_rank"] = int(np.argmax(counts[1:]) + 1)
        block["min_rank"] = int(np.min(ranks[:, idx]))
        block["max_rank"] = int(np.max(ranks[:, idx]))
        block["p_top_2"] = float(np.mean(ranks[:, idx] <= 2))
        block["p_top_3"] = float(np.mean(ranks[:, idx] <= 3))
        rank_frequency[method] = block

    pairwise: dict[str, dict[str, float]] = {}
    for i, left in enumerate(methods):
        row: dict[str, float] = {}
        for j, right in enumerate(methods):
            if i == j:
                continue
            row[right] = float(np.mean(scores[:, i] > scores[:, j]))
        pairwise[left] = row

    pair_products = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            ref_sign = np.sign(reference_scores[i] - reference_scores[j])
            sample_sign = np.sign(scores[:, i] - scores[:, j])
            pair_products.append(ref_sign * sample_sign)
    pair_products_arr = np.vstack(pair_products).T
    kendall_tau = np.mean(pair_products_arr, axis=1)

    rank_diff = ranks - reference_ranks[None, :]
    denom = n_methods * (n_methods**2 - 1)
    spearman = 1.0 - 6.0 * np.sum(rank_diff**2, axis=1) / denom

    mean_scores = np.mean(scores, axis=0)
    score_quantiles = {}
    for idx, method in enumerate(methods):
        qs = np.quantile(scores[:, idx], [0.05, 0.50, 0.95])
        score_quantiles[method] = {
            "mean": float(mean_scores[idx]),
            "q05": float(qs[0]),
            "median": float(qs[1]),
            "q95": float(qs[2]),
        }

    return {
        "reference_ranking": [
            {
                "rank": int(reference_ranks[idx]),
                "method": method,
                "score": float(reference_scores[idx]),
            }
            for idx, method in sorted(
                enumerate(methods),
                key=lambda item: reference_ranks[item[0]],
            )
        ],
        "rank_frequency": rank_frequency,
        "pairwise_win_probability": pairwise,
        "rank_correlation_with_reference": {
            "kendall_tau_mean": float(np.mean(kendall_tau)),
            "kendall_tau_std": float(np.std(kendall_tau)),
            "spearman_mean": float(np.mean(spearman)),
            "spearman_std": float(np.std(spearman)),
        },
        "score_quantiles": score_quantiles,
    }


def _weight_summary(weights: np.ndarray, names: tuple[str, ...]) -> dict:
    out = {}
    for idx, name in enumerate(names):
        qs = np.quantile(weights[:, idx], [0.05, 0.50, 0.95])
        out[name] = {
            "mean": float(np.mean(weights[:, idx])),
            "q05": float(qs[0]),
            "median": float(qs[1]),
            "q95": float(qs[2]),
        }
    return out


def _nlps_scores(components: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weights @ components.T


def _ter_scores(
    ter_components: np.ndarray,
    param_efficiency: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return (weights @ ter_components.T) * param_efficiency[None, :]


def _local_nlps_swap_thresholds(
    methods: list[str],
    components: np.ndarray,
    base_weights: np.ndarray,
) -> dict:
    """Find the nearest local-weight rank tie under NLPS priority constraints."""
    reference_scores = components @ base_weights
    order = list(np.argsort(-reference_scores))

    # Variables are w_NLC, w_1-BPPI, w_TER, w_NSS, epsilon.
    objective = np.array([0, 0, 0, 0, 1.0], dtype=float)
    a_eq = np.array([[1, 1, 1, 1, 0]], dtype=float)
    b_eq = np.array([1.0], dtype=float)

    a_ub: list[list[float]] = [
        [-1, 1, 0, 0, 0],  # w_NLC >= w_1-BPPI
        [0, -1, 1, 0, 0],  # w_1-BPPI >= w_TER
        [0, -1, 0, 1, 0],  # w_1-BPPI >= w_NSS
    ]
    b_ub: list[float] = [0.0, 0.0, 0.0]

    for idx in range(4):
        row = [0, 0, 0, 0, -1]
        row[idx] = 1
        a_ub.append(row)
        b_ub.append(float(base_weights[idx]))

        row = [0, 0, 0, 0, -1]
        row[idx] = -1
        a_ub.append(row)
        b_ub.append(float(-base_weights[idx]))

    bounds = [(0.0, 1.0)] * 4 + [(0.0, None)]

    thresholds: list[dict] = []
    for higher_pos, higher_idx in enumerate(order):
        for lower_idx in order[higher_pos + 1:]:
            row = list(components[higher_idx] - components[lower_idx]) + [0]
            result = linprog(
                objective,
                A_ub=np.array(a_ub + [row], dtype=float),
                b_ub=np.array(b_ub + [0.0], dtype=float),
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if not result.success:
                continue
            thresholds.append(
                {
                    "higher_at_published_weights": methods[higher_idx],
                    "lower_at_published_weights": methods[lower_idx],
                    "min_linf_epsilon_for_tie_or_reversal": float(result.fun),
                    "tie_weights": dict(
                        zip(NLPS_COMPONENTS, result.x[:4].astype(float).tolist())
                    ),
                }
            )

    thresholds.sort(key=lambda item: item["min_linf_epsilon_for_tie_or_reversal"])
    adjacent_pairs = []
    for higher_idx, lower_idx in zip(order[:-1], order[1:]):
        match = next(
            (
                item
                for item in thresholds
                if item["higher_at_published_weights"] == methods[higher_idx]
                and item["lower_at_published_weights"] == methods[lower_idx]
            ),
            None,
        )
        if match is not None:
            adjacent_pairs.append(match)

    return {
        "base_weights": dict(zip(NLPS_COMPONENTS, base_weights.tolist())),
        "interpretation": (
            "Minimum L-infinity perturbation from the published NLPS weights "
            "needed for the lower-ranked method to tie or overtake the higher-ranked "
            "method, subject to the same local priority constraints."
        ),
        "nearest_any_pair": thresholds[0] if thresholds else None,
        "all_pairs": thresholds,
        "adjacent_pairs": adjacent_pairs,
    }


def _summary_lines(payloads: dict[str, dict]) -> list[str]:
    lines = [
        "Weight-sensitivity analysis summary",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for title, payload in payloads.items():
        lines.append(title)
        lines.append("-" * len(title))
        corr = payload["analysis"]["rank_correlation_with_reference"]
        lines.append(
            "Kendall tau vs reference: "
            f"{corr['kendall_tau_mean']:.3f} +/- {corr['kendall_tau_std']:.3f}"
        )
        for row in payload["analysis"]["reference_ranking"]:
            method = row["method"]
            freq = payload["analysis"]["rank_frequency"][method]
            lines.append(
                f"{row['rank']}. {method}: "
                f"P(top-1)={freq['rank_1']:.3f}, "
                f"P(top-2)={freq['p_top_2']:.3f}, "
                f"rank band={freq['min_rank']}-{freq['max_rank']}"
            )
        lines.append("")
    return lines


def _build_payload(
    experiment: str,
    n_samples: int,
    elapsed_s: float,
    constraints: list[str],
    weight_names: tuple[str, ...],
    weights: np.ndarray,
    analysis: dict,
) -> dict:
    return {
        "experiment": experiment,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "elapsed_s": elapsed_s,
        "constraints": constraints,
        "weight_summary": _weight_summary(weights, weight_names),
        "analysis": analysis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--out-dir",
        default="results/weight_sensitivity",
        help="Directory for JSON and text outputs.",
    )
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--local-eps",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15],
        help="L-infinity perturbation radii around published NLPS weights.",
    )
    args = parser.parse_args()

    t_all = time.perf_counter()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    methods, nlps_components = _load_nlps_components(results_dir)
    ter_components, param_efficiency = _load_ter_components(results_dir, methods)
    reference_nlps = nlps_components @ NLPS_WEIGHTS
    reference_ter = (ter_components @ TER_WEIGHTS) * param_efficiency

    written_payloads: dict[str, dict] = {}

    t0 = time.perf_counter()
    global_weights = _sample_global_nlps_weights(
        rng,
        args.n_samples,
        args.batch_size,
    )
    global_scores = _nlps_scores(nlps_components, global_weights)
    payload = _build_payload(
        "global_admissible_nlps_weights",
        args.n_samples,
        time.perf_counter() - t0,
        [
            "w_NLC > w_1-BPPI",
            "w_1-BPPI >= w_TER",
            "w_1-BPPI >= w_NSS",
            "all weights >= 0.15",
            "sum weights = 1",
        ],
        NLPS_COMPONENTS,
        global_weights,
        _rank_summary(methods, global_scores, reference_nlps),
    )
    _write_json(out_dir / "nlps_global_sensitivity.json", payload)
    written_payloads["Global NLPS admissible weights"] = payload

    local_payload = {
        "experiment": "local_nlps_weight_perturbations",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples_per_epsilon": args.n_samples,
        "base_weights": dict(zip(NLPS_COMPONENTS, NLPS_WEIGHTS.tolist())),
        "local_swap_thresholds": _local_nlps_swap_thresholds(
            methods,
            nlps_components,
            NLPS_WEIGHTS,
        ),
        "constraint": [
            "max_i |w_i - w_i_base| <= epsilon",
            "w_NLC > w_1-BPPI",
            "w_1-BPPI >= w_TER",
            "w_1-BPPI >= w_NSS",
            "all weights >= 0",
            "sum weights = 1",
        ],
        "by_epsilon": {},
    }
    for epsilon in args.local_eps:
        t0 = time.perf_counter()
        local_weights = _sample_local_nlps_weights(
            rng,
            args.n_samples,
            epsilon,
            args.batch_size,
        )
        local_scores = _nlps_scores(nlps_components, local_weights)
        local_payload["by_epsilon"][str(epsilon)] = _build_payload(
            f"local_nlps_eps_{epsilon}",
            args.n_samples,
            time.perf_counter() - t0,
            local_payload["constraint"],
            NLPS_COMPONENTS,
            local_weights,
            _rank_summary(methods, local_scores, reference_nlps),
        )
    _write_json(out_dir / "nlps_local_sensitivity.json", local_payload)
    written_payloads["Local NLPS eps=0.10"] = local_payload["by_epsilon"]["0.1"]

    t0 = time.perf_counter()
    ter_weights = _sample_ter_weights(rng, args.n_samples, args.batch_size)
    varied_ter = _ter_scores(ter_components, param_efficiency, ter_weights)
    propagated_nlps = (
        NLPS_WEIGHTS[0] * nlps_components[:, 0][None, :]
        + NLPS_WEIGHTS[1] * nlps_components[:, 1][None, :]
        + NLPS_WEIGHTS[2] * varied_ter
        + NLPS_WEIGHTS[3] * nlps_components[:, 3][None, :]
    )
    ter_payload = {
        "experiment": "ter_internal_weight_sensitivity",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": args.n_samples,
        "elapsed_s": time.perf_counter() - t0,
        "constraints": [
            "w_curve_score > w_best_train_score > w_test_score",
            "w_curve_score >= 0.40",
            "w_test_score >= 0.10",
            "sum weights = 1",
        ],
        "weight_summary": _weight_summary(ter_weights, TER_COMPONENTS),
        "ter_analysis": _rank_summary(methods, varied_ter, reference_ter),
        "propagated_nlps_analysis": _rank_summary(
            methods,
            propagated_nlps,
            reference_nlps,
        ),
    }
    _write_json(out_dir / "ter_internal_sensitivity.json", ter_payload)
    written_payloads["TER internal weights"] = {
        "analysis": ter_payload["ter_analysis"]
    }
    written_payloads["NLPS with TER perturbed"] = {
        "analysis": ter_payload["propagated_nlps_analysis"]
    }

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "elapsed_s_total": time.perf_counter() - t_all,
        "outputs": [
            "nlps_global_sensitivity.json",
            "nlps_local_sensitivity.json",
            "ter_internal_sensitivity.json",
            "summary.txt",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)

    summary = "\n".join(_summary_lines(written_payloads))
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
