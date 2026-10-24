"""Ancilla + data re-uploading: repeated encoding on the system with CPTP coupling.

Each layer: (1) encode ``x`` on all system qubits via ``RY(x)`` (Fourier-style
re-upload), (2) trainable ``RY``/``RZ`` on the system, (3) system→ancilla
``CNOT``s and a small trainable rotation on the ancilla — same CPTP picture as
``ancilla_tracing`` but with per-layer data injection.

Weight tensor shape matches data re-uploading / ancilla tracing:
``(n_layers, n_qubits, 2)`` for system rotations; ancilla uses
``weights[layer, 0, 0]`` as in :func:`~src.circuits.ancilla_tracing.build_circuit`.
"""

import pennylane as qml
from pennylane import numpy as pnp


def build_circuit(n_qubits: int, n_layers: int):
    """Return ``circuit(weights, x) -> <Z_0>`` on ``default.mixed`` (n system + 1 ancilla)."""
    total_wires = n_qubits + 1
    ancilla_wire = n_qubits
    dev = qml.device("default.mixed", wires=total_wires)

    @qml.qnode(dev)
    def circuit(weights: pnp.ndarray, x: float) -> float:
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(x, wires=i)
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, ancilla_wire])
            qml.RY(weights[layer, 0, 0] * 0.5, wires=ancilla_wire)
        return qml.expval(qml.PauliZ(0))

    return circuit
