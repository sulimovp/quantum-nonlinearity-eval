"""PQK-style baselines to contrast with QNLEP scores.

These are **not** components of NLPS. They help position QNLEP next to common
practice (parameter counting, random-init output spread).

- **parameter_count**: product of ``param_shape`` (resource index).
- **readout_var_z0**: variance of ``<Z_0>(theta, x_fix)`` over uniform random
  ``theta`` at fixed ``x_fix``. This is a **lightweight** proxy for how much
  the readout moves under random parameters — related in spirit to output
  diversity used in expressibility discussions, but **not** the full Sim et al.
  expressibility (Haar KL integral).

Run :mod:`notebooks.11_comparison_metrics` (notebook 11) to write
``results/pqk_baseline_metrics.json`` for use in notebook 09.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from pennylane import numpy as pnp

from src.qnlep_registry import QNLEP_BUILDERS, param_shape


def parameter_count(method: str, n_qubits: int, n_layers: int) -> int:
    sh = param_shape(method, n_qubits, n_layers)
    return int(np.prod(sh))


def readout_spread_z0(
    method: str,
    n_qubits: int,
    n_layers: int,
    *,
    n_samples: int = 256,
    x_fix: float = 0.0,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    builder: Callable = QNLEP_BUILDERS[method]
    circuit = builder(n_qubits, n_layers)
    sh = param_shape(method, n_qubits, n_layers)
    ys: list[float] = []
    for _ in range(n_samples):
        w = pnp.array(rng.uniform(0, 2 * np.pi, sh))
        ys.append(float(circuit(w, x_fix)))
    arr = np.asarray(ys, dtype=float)
    return {
        "readout_mean_z0": float(np.mean(arr)),
        "readout_var_z0": float(np.var(arr)),
        "n_samples": float(n_samples),
        "x_fix": float(x_fix),
    }


def compute_all_baselines(
    method_order: list[str],
    n_qubits: int,
    n_layers: int,
    **kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for m in method_order:
        out[m] = {
            "parameter_count": float(parameter_count(m, n_qubits, n_layers)),
            **readout_spread_z0(m, n_qubits, n_layers, **kwargs),
        }
    return out
