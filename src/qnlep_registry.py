"""QNLEP: method names, circuit builders, and parameter tensor shapes.

Notebooks 03–04–07 use different weight layouts (e.g. QCNN uses (L, 4), QRC
uses (n_qubits, 2)). Use :func:`param_shape` when initialising optimisers or
sampling gradients for BPPI.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from pennylane import numpy as pnp

from src.circuits.ancilla_reuploading import build_circuit as build_ancilla_reupload
from src.circuits.ancilla_tracing import build_circuit as build_ancilla
from src.circuits.data_reuploading import build_circuit as build_reuploading
from src.circuits.qcnn import build_circuit as build_qcnn, param_shape as qcnn_param_shape
from src.circuits.quantum_reservoir import build_circuit as build_qrc, param_shape as qrc_param_shape
from src.circuits.strongly_entangling import build_circuit as build_sel, param_shape as sel_param_shape

Builder = Callable[..., Callable]

QNLEP_BUILDERS: Dict[str, Builder] = {
    "data_reuploading": build_reuploading,
    "ancilla_tracing": build_ancilla,
    "ancilla_reuploading": build_ancilla_reupload,
    "strongly_entangling": build_sel,
    "qcnn": build_qcnn,
    "quantum_reservoir": build_qrc,
}


def _shape_reupload_family(n_qubits: int, n_layers: int) -> Tuple[int, ...]:
    return (n_layers, n_qubits, 2)


def _shape_qrc(n_qubits: int, n_layers: int) -> Tuple[int, ...]:
    _ = n_layers
    return (n_qubits, 2)


QNLEP_PARAM_SHAPES: Dict[str, Callable[[int, int], Tuple[int, ...]]] = {
    "data_reuploading": _shape_reupload_family,
    "ancilla_tracing": _shape_reupload_family,
    "ancilla_reuploading": _shape_reupload_family,
    "strongly_entangling": lambda nq, nl: sel_param_shape(nq, nl),
    "qcnn": lambda nq, nl: qcnn_param_shape(nq, nl),
    "quantum_reservoir": _shape_qrc,
}


def param_shape(method: str, n_qubits: int, n_layers: int) -> Tuple[int, ...]:
    if method not in QNLEP_PARAM_SHAPES:
        raise KeyError(f"Unknown QNLEP method {method!r}; choose from {sorted(QNLEP_BUILDERS)}")
    return QNLEP_PARAM_SHAPES[method](n_qubits, n_layers)


def init_weights_normal(
    method: str,
    n_qubits: int,
    n_layers: int,
    scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> pnp.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    shape = param_shape(method, n_qubits, n_layers)
    return pnp.array(rng.normal(0, scale, shape), requires_grad=True)


def init_weights_uniform(
    method: str,
    n_qubits: int,
    n_layers: int,
    rng: np.random.Generator | None = None,
) -> pnp.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    shape = param_shape(method, n_qubits, n_layers)
    return pnp.array(rng.uniform(0, 2 * np.pi, shape), requires_grad=True)
