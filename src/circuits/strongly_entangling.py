"""Strongly Entangling Layers circuit: hardware-efficient ansatz baseline.

Architecture: angle-embedding of input x followed by StronglyEntanglingLayers
(parameterized Rot gates + ring of CNOTs, repeated n_layers times).
Non-linearity mechanism: encoding-based (same as re-uploading but data encoded
only once, expressiveness comes from entangling structure depth).

Reference:
  Schuld, Bocharov, Svore, Wiebe — "Circuit-centric quantum classifiers" (2020)
  PennyLane template: qml.StronglyEntanglingLayers
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def param_shape(n_qubits: int, n_layers: int) -> tuple:
    """Weight tensor shape required by build_circuit.

    StronglyEntanglingLayers uses 3 Euler angles per qubit per layer.
    Shape: (n_layers, n_qubits, 3).
    """
    return (n_layers, n_qubits, 3)


def build_circuit(n_qubits: int, n_layers: int) -> qml.QNode:
    """Build a Strongly Entangling Layers circuit.

    Args:
        n_qubits: number of qubits.
        n_layers: number of StronglyEntanglingLayers repetitions.

    Returns:
        qnode with signature circuit(weights, x) -> float.
        weights shape: (n_layers, n_qubits, 3)  — use param_shape() to initialise.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights: pnp.ndarray, x: float) -> float:
        # Encode input once on all qubits via angle embedding
        for w in range(n_qubits):
            qml.RY(x, wires=w)
        # Strongly entangling layers: Rot(alpha, beta, gamma) + CNOT ring
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit
