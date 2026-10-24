"""Quantum Reservoir Computing (QRC) circuit.

Architecture:
  Reservoir — a fixed (non-trainable) random strongly-entangling circuit
    seeded at construction time. This represents the "echo state" dynamics.
  Encoding — input x encoded via RY on all qubits before the reservoir.
  Readout — a small trainable layer (RY + RZ per qubit) applied after the
    fixed reservoir; only these weights are optimised.

Non-linearity mechanism: physical / structural — the fixed random unitary
creates a high-dimensional projection of the input; the readout exploits this
rich representation. Non-linearity is intrinsic to the random quantum dynamics
(unlike re-uploading which builds it via repeated encoding).

Key hyperparameter: reservoir_layers controls the depth of the fixed circuit,
which determines the effective "memory" and mixing of the reservoir.

Reference:
  Fujii & Nakajima — "Harnessing Disordered-Ensemble Quantum Dynamics
  for Machine Learning," Phys. Rev. Applied 8, 024030 (2017).
  Source: arxiv.org/abs/1602.08159
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def param_shape(n_qubits: int, n_layers: int) -> tuple:
    """Weight tensor shape required by build_circuit.

    Only the readout layer is trainable: one RY + one RZ per qubit.
    Shape: (n_qubits, 2).
    n_layers controls reservoir depth (not trainable parameter count).
    """
    return (n_qubits, 2)


def build_circuit(
    n_qubits: int,
    n_layers: int,
    reservoir_seed: int = 0,
    reservoir_layers: int = 3,
) -> qml.QNode:
    """Build a QRC circuit.

    Args:
        n_qubits: number of qubits.
        n_layers: unused (kept for API consistency); use reservoir_layers instead.
        reservoir_seed: random seed for fixing the reservoir circuit.
        reservoir_layers: depth of the fixed reservoir circuit.

    Returns:
        qnode with signature circuit(weights, x) -> float.
        weights shape: (n_qubits, 2) — use param_shape() to initialise.
        Only the readout weights are optimised; the reservoir is frozen.
    """
    # Fix the reservoir once at construction time
    rng = np.random.default_rng(reservoir_seed)
    reservoir_weights = pnp.tensor(
        rng.uniform(0, 2 * np.pi, qml.StronglyEntanglingLayers.shape(reservoir_layers, n_qubits)),
        requires_grad=False,
    )

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights: pnp.ndarray, x: float) -> float:
        # Encode input
        for w in range(n_qubits):
            qml.RY(x, wires=w)
        # Fixed reservoir (non-trainable)
        qml.StronglyEntanglingLayers(reservoir_weights, wires=range(n_qubits))
        # Trainable readout layer
        for w in range(n_qubits):
            qml.RY(weights[w, 0], wires=w)
            qml.RZ(weights[w, 1], wires=w)
        return qml.expval(qml.PauliZ(0))

    return circuit
