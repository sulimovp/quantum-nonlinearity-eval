"""Lorenz-task builders for the QNLEP circuit families.

These mirror the scalar-input QNLEP registry, but accept a 3D state vector
``state = (x_t, y_t, z_t)`` for one-step-ahead Lorenz prediction.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.circuits.qcnn import _conv_layer, _pool_layer


def _lorenz_encode(state, n_qubits: int) -> np.ndarray:
    """Map the 3D Lorenz state onto ``n_qubits`` encoding angles (cycle if wider)."""
    vec = np.asarray(state, dtype=float).ravel()
    if vec.size == 0:
        raise ValueError("Lorenz state must be non-empty")
    return np.array([float(vec[i % vec.size]) for i in range(n_qubits)], dtype=float)


def build_lorenz_data_reuploading(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, state):
        enc = _lorenz_encode(state, n_qubits)
        for layer in range(n_layers):
            for w in range(n_qubits):
                qml.RY(enc[w], wires=w)
            for w in range(n_qubits):
                qml.RY(weights[layer, w, 0], wires=w)
                qml.RZ(weights[layer, w, 1], wires=w)
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_lorenz_ancilla_tracing(n_qubits: int, n_layers: int):
    anc = n_qubits
    dev = qml.device("default.mixed", wires=n_qubits + 1)

    @qml.qnode(dev)
    def circuit(weights, state):
        enc = _lorenz_encode(state, n_qubits)
        for w in range(n_qubits):
            qml.RY(enc[w], wires=w)
        for layer in range(n_layers):
            for w in range(n_qubits):
                qml.RY(weights[layer, w, 0], wires=w)
                qml.RZ(weights[layer, w, 1], wires=w)
            for w in range(n_qubits):
                qml.CNOT(wires=[w, anc])
            qml.RY(weights[layer, 0, 0] * 0.5, wires=anc)
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_lorenz_ancilla_reuploading(n_qubits: int, n_layers: int):
    anc = n_qubits
    dev = qml.device("default.mixed", wires=n_qubits + 1)

    @qml.qnode(dev)
    def circuit(weights, state):
        enc = _lorenz_encode(state, n_qubits)
        for layer in range(n_layers):
            for w in range(n_qubits):
                qml.RY(enc[w], wires=w)
            for w in range(n_qubits):
                qml.RY(weights[layer, w, 0], wires=w)
                qml.RZ(weights[layer, w, 1], wires=w)
            for w in range(n_qubits):
                qml.CNOT(wires=[w, anc])
            qml.RY(weights[layer, 0, 0] * 0.5, wires=anc)
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_lorenz_strongly_entangling(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, state):
        enc = _lorenz_encode(state, n_qubits)
        for w in range(n_qubits):
            qml.RY(enc[w], wires=w)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_lorenz_qcnn(n_qubits: int, n_layers: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, state):
        enc = _lorenz_encode(state, n_qubits)
        for w in range(n_qubits):
            qml.RY(enc[w], wires=w)
        for layer in range(n_layers):
            _conv_layer(weights[layer], n_qubits)
            _pool_layer(weights[layer], n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_lorenz_quantum_reservoir(
    n_qubits: int,
    n_layers: int,
    reservoir_seed: int = 0,
    reservoir_layers: int | None = None,
):
    reservoir_layers = n_layers if reservoir_layers is None else reservoir_layers
    rng = np.random.default_rng(reservoir_seed)
    reservoir_weights = pnp.tensor(
        rng.uniform(0, 2 * np.pi, qml.StronglyEntanglingLayers.shape(reservoir_layers, n_qubits)),
        requires_grad=False,
    )
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, state):
        enc = _lorenz_encode(state, n_qubits)
        for w in range(n_qubits):
            qml.RY(enc[w], wires=w)
        qml.StronglyEntanglingLayers(reservoir_weights, wires=range(n_qubits))
        for w in range(n_qubits):
            qml.RY(weights[w, 0], wires=w)
            qml.RZ(weights[w, 1], wires=w)
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_lorenz_registry(n_qubits: int, n_layers: int, run_seed: int = 0):
    """Return the full QNLEP registry for Lorenz one-step prediction."""
    return {
        "data_reuploading": build_lorenz_data_reuploading(n_qubits, n_layers),
        "ancilla_tracing": build_lorenz_ancilla_tracing(n_qubits, n_layers),
        "ancilla_reuploading": build_lorenz_ancilla_reuploading(n_qubits, n_layers),
        "strongly_entangling": build_lorenz_strongly_entangling(n_qubits, n_layers),
        "qcnn": build_lorenz_qcnn(n_qubits, n_layers),
        "quantum_reservoir": build_lorenz_quantum_reservoir(
            n_qubits,
            n_layers,
            reservoir_seed=run_seed,
            reservoir_layers=n_layers,
        ),
    }
