"""
Shared paths and integer knobs for protocol notebooks and sweep scripts.

Default results root is the project ``results/`` directory (parent of ``src/``),
so script runs from the repo root and notebook runs from ``notebooks/`` both
land artifacts in the same place. Override with ``QNLEP_PROTOCOL_RESULTS`` or
``QNLEP_RESULTS_SUBDIR`` (e.g. ``sweeps/n3_L3/seed_42``) for sweeps.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_RESULTS = Path(__file__).resolve().parent.parent / "results"


def protocol_results_dir() -> Path:
    """Directory for protocol JSON artifacts (creates parents)."""
    sub = os.environ.get("QNLEP_RESULTS_SUBDIR", "").strip().replace("\\", "/")
    env = os.environ.get("QNLEP_PROTOCOL_RESULTS", "").strip()
    base = Path(env) if env else _REPO_RESULTS
    if sub:
        base = base / sub
    base.mkdir(parents=True, exist_ok=True)
    return base


def protocol_n_qubits(default: int = 3) -> int:
    return int(os.environ.get("QNLEP_N_QUBITS", str(default)))


def protocol_n_layers(default: int = 3) -> int:
    return int(os.environ.get("QNLEP_N_LAYERS", str(default)))
