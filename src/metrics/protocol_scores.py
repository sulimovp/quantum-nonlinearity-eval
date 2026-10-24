"""Shared scoring helpers for the QNLEP notebooks.

These helpers intentionally separate:
- raw diagnostics (R² curves, slope fits, learning curves), and
- bounded protocol scores used in tables / NLPS.

The goal is to avoid brittle threshold-only scalars while still exporting
enough information for reviewers to audit the underlying behaviour.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def clip01(value: float) -> float:
    """Clamp a numeric value into ``[0, 1]``."""
    return float(np.clip(value, 0.0, 1.0))


def shifted_r2_score(r2: float) -> float:
    """Map R² into a bounded skill score in ``[0, 1]``.

    ``R² = -1`` maps to ``0``, ``R² = 0`` maps to ``0.5``, and
    ``R² = 1`` maps to ``1``. Values below ``-1`` are clipped to ``0``.
    """
    if np.isnan(r2):
        return math.nan
    return clip01((float(r2) + 1.0) / 2.0)


def positive_r2_score(r2: float) -> float:
    """Clip R² into ``[0, 1]`` for quality-only aggregation."""
    if np.isnan(r2):
        return math.nan
    return clip01(float(r2))


def k_star_at_threshold(
    r2_values: Sequence[float],
    k_values: Sequence[int],
    threshold: float,
) -> int | None:
    """Largest ``k`` with ``R² >= threshold``; ``None`` if none pass."""
    passed = [int(k) for k, r2 in zip(k_values, r2_values) if float(r2) >= threshold]
    return max(passed) if passed else None


def continuous_frequency_score(
    r2_values: Sequence[float],
    k_values: Sequence[int],
) -> float:
    """Continuous NLC replacement based on area under fit-quality vs log-frequency.

    The score integrates positive-clipped R² over the normalized log2-frequency
    axis so near-misses at higher frequencies contribute smoothly without
    collapsing to ``0`` / ``null``.
    """
    if len(r2_values) != len(k_values):
        raise ValueError("r2_values and k_values must have the same length")
    if len(k_values) < 2:
        return positive_r2_score(float(r2_values[0])) if r2_values else math.nan

    xs = np.log2(np.asarray(k_values, dtype=float))
    denom = float(xs[-1] - xs[0])
    if denom <= 0:
        return math.nan
    xs = (xs - xs[0]) / denom
    ys = np.array([positive_r2_score(v) for v in r2_values], dtype=float)
    return float(np.trapezoid(ys, xs))


def smooth_bppi_from_slope(alpha: float, scale: float = 2.0) -> float:
    """Map a gradient-variance slope to a smooth BPPI score in ``[0, 1]``.

    ``alpha >= 0`` stays near ``0`` (healthy / flat or increasing variance).
    More negative slopes move smoothly toward ``1`` without hard clipping.
    """
    severity = max(-float(alpha), 0.0)
    return float(1.0 - np.exp(-scale * severity))


def noise_stability_score(
    r2_curve: Sequence[float],
    noise_levels: Sequence[float],
) -> float:
    """Continuous NSS based on bounded skill under noise.

    We integrate a shifted-and-clipped R² skill curve rather than raw R² or
    positive-clipped R² only. This keeps the metric bounded and avoids
    over-fragile exact zeros when performance is slightly below the mean
    predictor.
    """
    if len(r2_curve) != len(noise_levels):
        raise ValueError("r2_curve and noise_levels must have the same length")
    if not noise_levels:
        return math.nan
    xs = np.asarray(noise_levels, dtype=float)
    span = float(xs[-1] - xs[0])
    if span <= 0:
        return shifted_r2_score(float(r2_curve[0]))
    ys = np.array([shifted_r2_score(v) for v in r2_curve], dtype=float)
    return float(np.trapezoid(ys, xs) / span)


def param_efficiency_score(n_params: int, min_params: int) -> float:
    """Smaller parameter counts are rewarded smoothly, not brutally."""
    n_params = max(int(n_params), 1)
    min_params = max(int(min_params), 1)
    return float(math.sqrt(min_params / n_params))


def learning_curve_score(step_scores: Sequence[float]) -> float:
    """Normalized area under a bounded train-score curve."""
    if not step_scores:
        return math.nan
    ys = np.asarray(step_scores, dtype=float)
    xs = np.linspace(0.0, 1.0, len(ys))
    return float(np.trapezoid(ys, xs))


def trainability_efficiency_score(
    train_step_scores: Sequence[float],
    test_r2: float,
    n_params: int,
    min_params: int,
) -> dict[str, float]:
    """Continuous TER replacement built from speed, quality, and parameter cost.

    Returns a dictionary so notebooks can export the components alongside the
    final scalar.
    """
    if not train_step_scores:
        return {
            "curve_score": math.nan,
            "best_train_score": math.nan,
            "test_score": math.nan,
            "param_efficiency": math.nan,
            "ter_raw": math.nan,
        }

    curve_score = learning_curve_score(train_step_scores)
    best_train_score = float(np.max(train_step_scores))
    test_score = shifted_r2_score(float(test_r2))
    param_eff = param_efficiency_score(n_params, min_params)
    ter_raw = (0.5 * curve_score + 0.3 * best_train_score + 0.2 * test_score) * param_eff
    return {
        "curve_score": float(curve_score),
        "best_train_score": float(best_train_score),
        "test_score": float(test_score),
        "param_efficiency": float(param_eff),
        "ter_raw": float(ter_raw),
    }


def summarize_r2_curve(values: Iterable[float]) -> dict[str, float]:
    """Compact summary for exported metadata."""
    arr = np.asarray(list(values), dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "final": float(arr[-1]),
    }
