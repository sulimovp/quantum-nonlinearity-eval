"""Ancilla tracing circuit: Non-linear maps via CPTP operations (partial trace)."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def partial_trace(rho_full, trace_qubits, n_qubits):
    """Compute partial trace over specified qubits.

    Traces out qubits in trace_qubits from a multi-qubit density matrix,
    returning the reduced density matrix on the remaining qubits.

    Args:
        rho_full: full density matrix (2^n × 2^n array)
        trace_qubits: set/list of qubit indices to trace out
        n_qubits: total number of qubits

    Returns:
        reduced density matrix (2^(n-len(trace_qubits)) × 2^(n-len(trace_qubits)))
    """
    # Convert to numpy for reliable computation
    rho_np = np.asarray(rho_full)

    keep_qubits = [i for i in range(n_qubits) if i not in trace_qubits]
    n_keep = len(keep_qubits)
    dim_keep = 2 ** n_keep
    dim_full = 2 ** n_qubits

    rho_reduced = np.zeros((dim_keep, dim_keep), dtype=complex)

    for i_full in range(dim_full):
        for j_full in range(dim_full):
            # Convert indices to binary
            i_bits = [(i_full >> k) & 1 for k in range(n_qubits)]
            j_bits = [(j_full >> k) & 1 for k in range(n_qubits)]

            # Extract bits for keep and trace qubits
            i_keep = tuple(i_bits[k] for k in keep_qubits)
            j_keep = tuple(j_bits[k] for k in keep_qubits)
            i_trace = tuple(i_bits[k] for k in trace_qubits)
            j_trace = tuple(j_bits[k] for k in trace_qubits)

            # Only add to reduced matrix if trace bits are equal
            if i_trace == j_trace:
                i_keep_idx = sum(i_keep[p] * (2**p) for p in range(len(keep_qubits)))
                j_keep_idx = sum(j_keep[p] * (2**p) for p in range(len(keep_qubits)))
                rho_reduced[i_keep_idx, j_keep_idx] += rho_np[i_full, j_full]

    return rho_reduced


def build_circuit(n_qubits: int, n_layers: int):
    """Build an ancilla-tracing circuit using CPTP maps (partial trace).

    Uses n_qubits system qubits + 1 ancilla qubit. Applies a unitary to the combined
    system, then traces out the ancilla to obtain a non-unitary CPTP map acting on
    the system state. This introduces non-linearity structurally via the partial trace.

    Data x is encoded once on system qubits (no repeated re-uploading).

    Returns:
        A function that takes (weights, x) and returns the expectation value
        <Z_0> computed from the reduced density matrix after partial tracing.
    """
    total_wires = n_qubits + 1
    ancilla_wire = n_qubits
    dev = qml.device("default.mixed", wires=total_wires)

    @qml.qnode(dev)
    def get_full_density_matrix(weights: pnp.ndarray, x: float):
        """Apply unitary on system + ancilla and return full density matrix."""
        # Encode data once on system qubits (no re-uploading)
        for i in range(n_qubits):
            qml.RY(x, wires=i)

        # Apply trainable unitary on system + ancilla
        for layer in range(n_layers):
            # Trainable gates on system qubits
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            # Entangle system with ancilla via CNOT (no re-uploading of x)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, ancilla_wire])

            # Trainable rotation on ancilla
            qml.RY(weights[layer, 0, 0] * 0.5, wires=ancilla_wire)

        # Return full density matrix before partial trace
        return qml.density_matrix(wires=list(range(total_wires)))

    @qml.qnode(dev)
    def circuit_implicit(weights: pnp.ndarray, x: float) -> float:
        """Implicit partial trace via marginal expectation (differentiable)."""
        # Encode data once on system qubits (no re-uploading)
        for i in range(n_qubits):
            qml.RY(x, wires=i)

        # Apply trainable unitary on system + ancilla
        for layer in range(n_layers):
            # Trainable gates on system qubits
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

            # Entangle system with ancilla via CNOT (no re-uploading of x)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, ancilla_wire])

            # Trainable rotation on ancilla
            qml.RY(weights[layer, 0, 0] * 0.5, wires=ancilla_wire)

        # Measure expectation on qubit 0 only
        # The ancilla is implicitly traced out (marginalized)
        return qml.expval(qml.PauliZ(0))

    # Return the implicit (differentiable) circuit for training
    # Attach the full density matrix QNode for inspection/visualization
    circuit_implicit.qnode_full = get_full_density_matrix
    circuit_implicit.partial_trace = partial_trace

    return circuit_implicit
