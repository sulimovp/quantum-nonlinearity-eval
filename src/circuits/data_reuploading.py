"""Data re-uploading circuit: Fourier expansion via repeated encoding."""

import pennylane as qml
from pennylane import numpy as pnp


def build_circuit(n_qubits: int, n_layers: int) -> qml.QNode:
    """Build a data re-uploading circuit.

    Repeatedly encodes input x interleaved with trainable rotations and
    entangling gates. Output is expectation of PauliZ on wire 0.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights: pnp.ndarray, x: float) -> float:
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(x, wires=i)
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    return circuit
