"""Depolarizing-noise versions of QNLEP circuits (default.mixed).

Mirrors the *clean* gate order in ``data_reuploading``, ``ancilla_tracing``,
``ancilla_reuploading``, ``strongly_entangling``, ``qcnn``, and ``quantum_reservoir``.
Ancilla-style circuits use ``n_qubits`` system wires + one ancilla on wire ``n_qubits``.
"""

from __future__ import annotations

import functools
from typing import Callable, Dict

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.circuits.qcnn import _conv_layer, _pool_layer


def _maybe_depolarize(p: float, wires) -> None:
    if p <= 0:
        return
    if isinstance(wires, int):
        wires = [wires]
    for w in wires:
        qml.DepolarizingChannel(p, wires=w)


def make_noisy_data_reuploading(noise_p: float, n_qubits: int, n_layers: int):
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, x):
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RY(x, wires=w)
                qml.RY(weights[l, w, 0], wires=w)
                qml.RZ(weights[l, w, 1], wires=w)
                _maybe_depolarize(noise_p, [w])
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
                _maybe_depolarize(noise_p, [w, w + 1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_noisy_ancilla_tracing(noise_p: float, n_qubits: int, n_layers: int):
    """Matches clean ancilla: single ``RY(x)`` on system, then layered unitary + ancilla."""
    dev = qml.device("default.mixed", wires=n_qubits + 1)
    anc = n_qubits

    @qml.qnode(dev)
    def circuit(weights, x):
        for w in range(n_qubits):
            qml.RY(x, wires=w)
            _maybe_depolarize(noise_p, [w])
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RY(weights[l, w, 0], wires=w)
                qml.RZ(weights[l, w, 1], wires=w)
                _maybe_depolarize(noise_p, [w])
            for w in range(n_qubits):
                qml.CNOT(wires=[w, anc])
                _maybe_depolarize(noise_p, [w, anc])
            qml.RY(weights[l, 0, 0] * 0.5, wires=anc)
            _maybe_depolarize(noise_p, [anc])
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_noisy_ancilla_reuploading(noise_p: float, n_qubits: int, n_layers: int):
    dev = qml.device("default.mixed", wires=n_qubits + 1)
    anc = n_qubits

    @qml.qnode(dev)
    def circuit(weights, x):
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RY(x, wires=w)
                _maybe_depolarize(noise_p, [w])
            for w in range(n_qubits):
                qml.RY(weights[l, w, 0], wires=w)
                qml.RZ(weights[l, w, 1], wires=w)
                _maybe_depolarize(noise_p, [w])
            for w in range(n_qubits):
                qml.CNOT(wires=[w, anc])
                _maybe_depolarize(noise_p, [w, anc])
            qml.RY(weights[l, 0, 0] * 0.5, wires=anc)
            _maybe_depolarize(noise_p, [anc])
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_noisy_strongly_entangling(noise_p: float, n_qubits: int, n_layers: int):
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, x):
        for w in range(n_qubits):
            qml.RY(x, wires=w)
            _maybe_depolarize(noise_p, [w])
        for l in range(n_layers):
            qml.StronglyEntanglingLayers(weights[l : l + 1], wires=range(n_qubits))
            _maybe_depolarize(noise_p, list(range(n_qubits)))
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_noisy_qcnn(noise_p: float, n_qubits: int, n_layers: int):
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, x):
        for w in range(n_qubits):
            qml.RY(x, wires=w)
            _maybe_depolarize(noise_p, [w])
        for l in range(n_layers):
            _conv_layer(weights[l], n_qubits)
            _maybe_depolarize(noise_p, list(range(n_qubits)))
            _pool_layer(weights[l], n_qubits)
            _maybe_depolarize(noise_p, list(range(n_qubits)))
        return qml.expval(qml.PauliZ(0))

    return circuit


def make_noisy_quantum_reservoir(
    noise_p: float,
    n_qubits: int,
    n_layers: int,
    reservoir_seed: int = 0,
    reservoir_layers: int = 3,
):
    rng = np.random.default_rng(reservoir_seed)
    rw = pnp.tensor(
        rng.uniform(0, 2 * np.pi, qml.StronglyEntanglingLayers.shape(reservoir_layers, n_qubits)),
        requires_grad=False,
    )
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, x):
        for w in range(n_qubits):
            qml.RY(x, wires=w)
            _maybe_depolarize(noise_p, [w])
        qml.StronglyEntanglingLayers(rw, wires=range(n_qubits))
        _maybe_depolarize(noise_p, list(range(n_qubits)))
        for w in range(n_qubits):
            qml.RY(weights[w, 0], wires=w)
            qml.RZ(weights[w, 1], wires=w)
            _maybe_depolarize(noise_p, [w])
        return qml.expval(qml.PauliZ(0))

    return circuit


NOISY_QNLEP_FACTORIES = {
    "data_reuploading": make_noisy_data_reuploading,
    "ancilla_tracing": make_noisy_ancilla_tracing,
    "ancilla_reuploading": make_noisy_ancilla_reuploading,
    "strongly_entangling": make_noisy_strongly_entangling,
    "qcnn": make_noisy_qcnn,
    "quantum_reservoir": make_noisy_quantum_reservoir,
}


def bound_noisy_factories(
    n_qubits: int,
    n_layers: int,
    reservoir_seed: int = 0,
    reservoir_layers: int | None = None,
) -> Dict[str, Callable]:
    """Return ``name -> f(p_noise) -> circuit(weights, x)`` with ``(n_qubits, n_layers)`` fixed.

    Notebook 07 calls ``f(p)`` only; raw factories require
    ``(p_noise, n_qubits, n_layers, ...)``.
    """
    out = {}
    for name, fn in NOISY_QNLEP_FACTORIES.items():
        kwargs = {"n_qubits": n_qubits, "n_layers": n_layers}
        if name == "quantum_reservoir":
            kwargs["reservoir_seed"] = reservoir_seed
            kwargs["reservoir_layers"] = n_layers if reservoir_layers is None else reservoir_layers
        out[name] = functools.partial(fn, **kwargs)
    return out
