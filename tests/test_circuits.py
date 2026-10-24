"""Tests for circuit implementations."""

import numpy as np
from pennylane import numpy as pnp

from src.circuits.data_reuploading import build_circuit
from src.circuits.ancilla_reuploading import build_circuit as build_aru
from src.qnlep_registry import param_shape, QNLEP_BUILDERS


def test_data_reuploading_output_shape() -> None:
    """Data re-uploading circuit returns scalar expectation."""
    circuit = build_circuit(n_qubits=2, n_layers=2)
    weights = pnp.array(np.zeros((2, 2, 2)))
    out = circuit(weights, 0.0)
    assert out.shape == ()
    assert -1 <= out <= 1


def test_ancilla_reuploading_output_shape() -> None:
    circuit = build_aru(n_qubits=2, n_layers=2)
    weights = pnp.array(np.zeros((2, 2, 2)))
    out = circuit(weights, 0.0)
    assert out.shape == ()
    assert -1 <= float(out) <= 1


def test_qnlep_registry_shapes_match_builders() -> None:
    nq, nl = 3, 2
    for name, builder in QNLEP_BUILDERS.items():
        sh = param_shape(name, nq, nl)
        w = pnp.array(np.zeros(sh))
        c = builder(nq, nl)
        out = c(w, 0.1)
        assert out.shape == ()
        assert -1 <= float(out) <= 1
