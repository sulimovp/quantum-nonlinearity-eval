#!/usr/bin/env python3
"""Headless real-data protocol (notebook 10 logic without Jupyter or figure cells).

Writes ``real_data_scores.json`` and ``real_data_ter_scores.json`` under the same
``QNLEP_RESULTS_SUBDIR`` convention as ``src/qnlep_run_paths.py``.

Run from repo root::

  python scripts/real_data_headless.py

Matches ``notebooks/10_real_data.ipynb`` training/export cells; keep in sync when
the notebook changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score, r2_score

# Repo root (notebooks use ".."; scripts live under scripts/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import (
    load_circles,
    load_iris_binary,
    load_mackey_glass,
    load_moons,
    load_regression_1d,
)
from src.qnlep_run_paths import protocol_n_layers, protocol_n_qubits, protocol_results_dir

from src.circuits.ancilla_reuploading import build_circuit as build_ancilla_reupload
from src.circuits.ancilla_tracing import build_circuit as build_ancilla
from src.circuits.data_reuploading import build_circuit as build_reuploading
from src.circuits.qcnn import build_circuit as build_qcnn
from src.circuits.qcnn import param_shape as qcnn_shape
from src.circuits.quantum_reservoir import build_circuit as build_qrc
from src.circuits.quantum_reservoir import param_shape as qrc_shape
from src.circuits.strongly_entangling import build_circuit as build_sel
from src.circuits.strongly_entangling import param_shape as sel_shape


def qnlep_env_flag(name: str) -> bool:
    v = os.environ.get(name)
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="Overrides QNLEP_SEED.")
    args = ap.parse_args()

    if args.seed is not None:
        os.environ["QNLEP_SEED"] = str(args.seed)

    run_seed = int(os.environ.get("QNLEP_SEED", "42"))
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    n_qubits = protocol_n_qubits(4)
    n_layers = protocol_n_layers(3)
    n_steps = int(os.environ.get("QNLEP_REAL_DATA_N_STEPS", "72"))
    lr = 0.05

    def default_shape(n_q: int, n_l: int):
        return (n_l, n_q, 2)

    method_registry: dict = {
        "data_reuploading": (build_reuploading, default_shape),
        "ancilla_tracing": (build_ancilla, default_shape),
        "ancilla_reuploading": (build_ancilla_reupload, default_shape),
        "strongly_entangling": (build_sel, sel_shape),
        "qcnn": (build_qcnn, qcnn_shape),
        "quantum_reservoir": (build_qrc, qrc_shape),
    }

    if qnlep_env_flag("QNLEP_REAL_DATA_FAST"):
        if "QNLEP_REAL_DATA_N_STEPS" not in os.environ:
            n_steps = min(n_steps, 22)
        _slow = ("quantum_reservoir", "qcnn", "strongly_entangling")
        method_registry = {k: v for k, v in method_registry.items() if k not in _slow}

    _ms = os.environ.get("QNLEP_REAL_DATA_METHODS", "").strip()
    if _ms:
        _allow_m = {x.strip() for x in _ms.split(",") if x.strip()}
        method_registry = {k: v for k, v in method_registry.items() if k in _allow_m}

    circuits: dict = {}
    for name, (builder, shape_fn) in method_registry.items():
        circuit = builder(n_qubits, n_layers)
        shape = shape_fn(n_qubits, n_layers)
        circuits[name] = (circuit, shape)
        print(f"  {name:25s}  param shape={shape}  n_params={int(np.prod(shape))}")

    if qnlep_env_flag("QNLEP_REAL_DATA_FAST") and "QNLEP_REAL_DATA_NSAMPLES" not in os.environ:
        _ns_default = "100"
    else:
        _ns_default = "160"
    _ns = int(os.environ.get("QNLEP_REAL_DATA_NSAMPLES", _ns_default))

    datasets = {
        "circles": load_circles(n_samples=_ns),
        "moons": load_moons(n_samples=_ns),
        "iris_1v2": load_iris_binary(classes=(1, 2)),
        "mackey_glass": load_mackey_glass(
            n_train=int(os.environ.get("QNLEP_REAL_DATA_MG_TRAIN", "120")),
            n_test=int(os.environ.get("QNLEP_REAL_DATA_MG_TEST", "60")),
        ),
        "x_sin_x": load_regression_1d(n_train=60, n_test=40),
    }

    task_map = {
        "circles": "classification",
        "moons": "classification",
        "iris_1v2": "classification",
        "mackey_glass": "regression",
        "x_sin_x": "regression",
    }

    _ds_filter = os.environ.get("QNLEP_REAL_DATA_DATASETS", "").strip()
    if qnlep_env_flag("QNLEP_REAL_DATA_FAST") and not _ds_filter:
        _keep = {"circles", "moons"}
    elif _ds_filter:
        _keep = {x.strip() for x in _ds_filter.split(",") if x.strip()}
    else:
        _keep = None
    if _keep is not None:
        datasets = {k: v for k, v in datasets.items() if k in _keep}
        task_map = {k: task_map[k] for k in datasets}

    # --- training helpers (notebook cells 5, 7, 8) ---
    def bce_loss(preds, y):
        probs = (pnp.tanh(preds) + 1) / 2
        probs = pnp.clip(probs, 1e-6, 1 - 1e-6)
        return -pnp.mean(y * pnp.log(probs) + (1 - y) * pnp.log(1 - probs))

    def train_circuit(circuit, param_shape, x_train, y_train, task, n_steps_i, lr_i):
        weights = pnp.array(np.random.normal(0, 0.1, param_shape), requires_grad=True)
        loss_fn = (
            bce_loss
            if task == "classification"
            else (lambda p, y: pnp.mean((p - y) ** 2))
        )

        def cost(w):
            preds = pnp.array(
                [circuit(w, float(x_train[i, 0])) for i in range(len(x_train))]
            )
            return loss_fn(preds, y_train)

        opt = qml.AdamOptimizer(stepsize=lr_i)
        history = []
        steps_to_thresh = n_steps_i
        for step in range(n_steps_i):
            weights, loss = opt.step_and_cost(cost, weights)
            history.append(float(loss))
            if steps_to_thresh == n_steps_i and step > 5:
                preds = np.array(
                    [
                        float(circuit(weights, float(x_train[i, 0])))
                        for i in range(len(x_train))
                    ]
                )
                if task == "classification":
                    acc = accuracy_score(y_train, (preds > 0).astype(int))
                    if acc >= 0.90:
                        steps_to_thresh = step + 1
                else:
                    r2 = r2_score(y_train, preds)
                    if r2 >= 0.80:
                        steps_to_thresh = step + 1

        return np.array(weights), history, steps_to_thresh

    def evaluate_circuit(circuit, weights, x, y, task):
        preds = np.array([float(circuit(weights, float(x[i, 0]))) for i in range(len(x))])
        if task == "classification":
            return accuracy_score(y, (preds > 0).astype(int))
        return r2_score(y, preds)

    class MLP(nn.Module):
        def __init__(self, in_features=1, hidden=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x)

    def train_mlp(x_train, y_train, task, n_steps_i, lr_i):
        model = MLP(in_features=x_train.shape[1], hidden=16)
        opt = torch.optim.Adam(model.parameters(), lr=lr_i)
        xtr = torch.tensor(x_train, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
        for _ in range(n_steps_i):
            opt.zero_grad()
            loss = loss_fn(model(xtr), ytr)
            loss.backward()
            opt.step()
        return model

    def eval_mlp(model, x, y, task):
        with torch.no_grad():
            preds = model(torch.tensor(x, dtype=torch.float32)).numpy().ravel()
        if task == "classification":
            return accuracy_score(y, (preds > 0).astype(int))
        return r2_score(y, preds)

    _dev_h = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(_dev_h, interface="torch", diff_method="backprop")
    def hybrid_quantum_features(x, q_weights):
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RY(x, wires=w)
                qml.RY(q_weights[l, w, 0], wires=w)
                qml.RZ(q_weights[l, w, 1], wires=w)
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    class HybridHQNN(nn.Module):
        def __init__(self, hidden: int = 16):
            super().__init__()
            self.q_weights = nn.Parameter(
                torch.tensor(
                    np.random.normal(0, 0.1, (n_layers, n_qubits, 2)),
                    dtype=torch.float32,
                )
            )
            self.head = nn.Sequential(
                nn.Linear(n_qubits, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x_batch):
            feats = torch.stack(
                [
                    torch.stack(hybrid_quantum_features(x_batch[i, 0], self.q_weights)).float()
                    for i in range(x_batch.shape[0])
                ]
            )
            return self.head(feats)

    def train_hqnn(x_train, y_train, task, n_steps_i, lr_i):
        model = HybridHQNN(hidden=16)
        opt = torch.optim.Adam(model.parameters(), lr=lr_i)
        xtr = torch.tensor(x_train, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
        for _ in range(n_steps_i):
            opt.zero_grad()
            loss = loss_fn(model(xtr), ytr)
            loss.backward()
            opt.step()
        return model

    def eval_hqnn(model, x, y, task):
        xe = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            preds = model(xe).numpy().ravel()
        if task == "classification":
            return accuracy_score(y, (preds > 0).astype(int))
        return r2_score(y, preds)

    def cap_train(x, y):
        cap = os.environ.get("QNLEP_REAL_DATA_MAX_TRAIN", "").strip()
        if (
            not cap
            and qnlep_env_flag("QNLEP_REAL_DATA_FAST")
            and "QNLEP_REAL_DATA_MAX_TRAIN" not in os.environ
        ):
            cap = "64"
        if not cap:
            return x, y
        n = int(cap)
        if len(y) <= n:
            return x, y
        rng = np.random.default_rng(run_seed)
        idx = rng.choice(len(y), size=n, replace=False)
        return x[idx], y[idx]

    # --- main evaluation loop ---
    all_results: dict = {}
    dataset_names = list(datasets.keys())

    for ds_name, (xtr, xte, ytr, yte) in datasets.items():
        task = task_map[ds_name]
        all_results[ds_name] = {}
        xtr_u, ytr_u = cap_train(xtr, ytr)
        print(f"\n=== {ds_name} ({task}) ===")

        mlp = train_mlp(xtr_u, ytr_u, task, n_steps, lr)
        mlp_score = eval_mlp(mlp, xte, yte, task)
        all_results[ds_name]["classical_mlp"] = {"score": mlp_score, "history": []}
        metric = "acc" if task == "classification" else "R²"
        print(f"  classical_mlp          {metric}={mlp_score:.3f}")

        x1_tr = xtr_u[:, :1]
        x1_te = xte[:, :1]
        hqnn = train_hqnn(x1_tr, ytr_u, task, n_steps, lr)
        hq_score = eval_hqnn(hqnn, x1_te, yte, task)
        all_results[ds_name]["hybrid_hqnn"] = {"score": hq_score, "history": []}
        print(f"  hybrid_hqnn            {metric}={hq_score:.3f}")

        for method_name, (circuit, p_shape) in circuits.items():
            weights, history, steps = train_circuit(
                circuit, p_shape, xtr_u, ytr_u, task, n_steps, lr
            )
            score = evaluate_circuit(circuit, weights, xte, yte, task)
            n_params = int(np.prod(p_shape))
            _s = float(max(1, int(steps)))
            _np = float(max(1, int(n_params)))
            ter_raw = 1.0 / (_s * _np)
            if int(steps) < int(n_steps):
                ter_raw *= float(
                    1.0 + 0.35 * (int(n_steps) - int(steps)) / max(float(n_steps), 1.0)
                )

            all_results[ds_name][method_name] = {
                "score": score,
                "history": history,
                "steps_to_thresh": steps,
                "n_params": n_params,
                "ter_raw": ter_raw,
                "circuit_weights": np.asarray(weights, dtype=float),
            }
            print(
                f"  {method_name:25s}  {metric}={score:.3f}  steps={steps:3d}  "
                f"n_params={n_params}"
            )

    results_dir = protocol_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    methods_q = list(circuits.keys())
    focus_ds = (
        "circles"
        if "circles" in all_results
        else "moons"
        if "moons" in all_results
        else next(iter(all_results))
    )
    ter_scores = {
        m: all_results[focus_ds][m].get("ter_raw", 0.0) for m in methods_q
    }
    max_ter = max(ter_scores.values()) or 1.0
    ter_scores_norm = {m: ter_scores[m] / max_ter for m in methods_q}

    with open(results_dir / "real_data_ter_scores.json", "w", encoding="utf-8") as f:
        json.dump(ter_scores_norm, f, indent=2)

    export = {
        ds: {
            m: {
                "score": all_results[ds][m]["score"],
                "steps": all_results[ds][m].get("steps_to_thresh", n_steps),
            }
            for m in all_results[ds]
        }
        for ds in all_results
    }
    with open(results_dir / "real_data_scores.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print("\nSaved real_data_ter_scores.json and real_data_scores.json")
    print(f"RESULTS_DIR={results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
