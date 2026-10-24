"""Compute notebook-03 NLC sweep for one method.

This helper mirrors the notebook-03 training recipe so expensive Fourier sweeps
can be precomputed per method in parallel and then merged back into
``results/nlc_results_partial.json``. It uses the same task-local seed strategy
as the notebook.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import r2_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qnlep_registry import QNLEP_BUILDERS, param_shape  # noqa: E402


K_VALUES = [1, 2, 4, 8, 16]
N_TRAIN = 36
N_EVAL = 120
N_STEPS = 50
N_STEPS_BY_METHOD = {
    "ancilla_reuploading": 100,
    "quantum_reservoir": 120,
}
LR = 0.1
N_QUBITS = 3
N_LAYERS = 3
QNLEP_METHOD_ORDER = [
    "data_reuploading",
    "ancilla_tracing",
    "ancilla_reuploading",
    "strongly_entangling",
    "qcnn",
    "quantum_reservoir",
]


def task_seed(run_seed: int, method_name: str, k: int) -> int:
    method_idx = QNLEP_METHOD_ORDER.index(method_name)
    return int(run_seed) * 10_000 + method_idx * 100 + int(k)


def train_on_task(method_name: str, circuit, x_tr, y_tr, run_seed: int, k: int):
    shape = param_shape(method_name, N_QUBITS, N_LAYERS)
    rng = np.random.default_rng(task_seed(run_seed, method_name, k))
    weights = pnp.array(rng.normal(0, 0.1, shape), requires_grad=True)

    def cost(w):
        preds = pnp.array([circuit(w, float(x)) for x in x_tr])
        return pnp.mean((preds - y_tr) ** 2)

    opt = qml.AdamOptimizer(stepsize=LR)
    history = []
    for _ in range(N_STEPS_BY_METHOD.get(method_name, N_STEPS)):
        weights, loss = opt.step_and_cost(cost, weights)
        history.append(float(loss))
    return weights, history


def build_circuit(method_name: str, run_seed: int):
    if method_name == "quantum_reservoir":
        return QNLEP_BUILDERS[method_name](N_QUBITS, N_LAYERS, reservoir_seed=run_seed)
    return QNLEP_BUILDERS[method_name](N_QUBITS, N_LAYERS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=QNLEP_METHOD_ORDER)
    parser.add_argument("--run-seed", type=int, default=42)
    args = parser.parse_args()

    x_train = np.linspace(-np.pi, np.pi, N_TRAIN)
    x_eval = np.linspace(-np.pi, np.pi, N_EVAL)
    circuit = build_circuit(args.method, args.run_seed)
    out = {}

    for k in K_VALUES:
        y_tr = np.sin(k * x_train)
        y_ev = np.sin(k * x_eval)
        weights, history = train_on_task(args.method, circuit, x_train, y_tr, args.run_seed, k)
        preds = np.array([float(circuit(weights, float(x))) for x in x_eval])
        r2 = float(r2_score(y_ev, preds))
        out[k] = {"r2": r2, "preds": preds.tolist(), "history": history}
        print(args.method, k, r2, flush=True)

    out_path = ROOT / "results" / f"nlc_method_{args.method}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("DONE", args.method, out_path, flush=True)


if __name__ == "__main__":
    main()
