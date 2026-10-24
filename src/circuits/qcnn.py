"""Quantum Convolutional Neural Network (QCNN) circuit.

Architecture (translation-invariant, no mid-circuit measurements):
  For each layer:
    Convolutional sublayer — same parameterized RY-CNOT-RY gate applied to all
      adjacent pairs (even-odd, then odd-even) with shared weights.
    Pooling sublayer — parameterized RZ + CNOT from even to odd qubits,
      compressing information downward through the register.
  Readout: expectation of PauliZ on qubit 0.

Non-linearity mechanism: structural — convolutional weight sharing + pooling
creates an effective hierarchical non-linear map (analogous to classical CNN).
The non-linearity is not from repeated encoding or ancilla tracing but from
the geometric structure of information flow.

Reference:
  Cong, Choi, Lukin — "Quantum convolutional neural networks" (2019)
  Nature Physics 15, 1273–1278.
  Source: arxiv.org/abs/1810.03912
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def param_shape(n_qubits: int, n_layers: int) -> tuple:
    """Weight tensor shape required by build_circuit.

    Each layer uses 4 shared parameters:
      [0] RY angle before CNOT on first wire  (conv)
      [1] RY angle before CNOT on second wire (conv)
      [2] RZ angle on even wire               (pool)
      [3] RZ angle on odd wire after CNOT     (pool)
    Shape: (n_layers, 4).
    """
    return (n_layers, 4)


def _conv_layer(params, n_qubits: int) -> None:
    """Translation-invariant convolutional sublayer.

    Applies the same 2-qubit gate RY-CNOT-RY to all adjacent pairs.
    Two sweeps (even-starting, odd-starting) for full coverage.
    """
    for start in (0, 1):
        for i in range(start, n_qubits - 1, 2):
            qml.RY(params[0], wires=i)
            qml.RY(params[1], wires=i + 1)
            qml.CNOT(wires=[i, i + 1])


def _pool_layer(params, n_qubits: int) -> None:
    """Pooling sublayer: RZ + CNOT pairs compress even→odd qubits."""
    for i in range(0, n_qubits - 1, 2):
        qml.RZ(params[2], wires=i)
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(params[3], wires=i + 1)


def build_circuit(n_qubits: int, n_layers: int) -> qml.QNode:
    """Build a QCNN circuit.

    Args:
        n_qubits: number of qubits (works best with powers of 2).
        n_layers: number of conv-pool block repetitions.

    Returns:
        qnode with signature circuit(weights, x) -> float.
        weights shape: (n_layers, 4) — use param_shape() to initialise.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights: pnp.ndarray, x: float) -> float:
        # Input encoding
        for w in range(n_qubits):
            qml.RY(x, wires=w)
        # Conv-pool blocks
        for l in range(n_layers):
            _conv_layer(weights[l], n_qubits)
            _pool_layer(weights[l], n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit
