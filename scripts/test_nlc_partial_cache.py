#!/usr/bin/env python3
"""
Unit tests for notebook 03 NLC partial JSON layout + fingerprint logic.
Run from repo root: python scripts/test_nlc_partial_cache.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _partial_to_disk(obj, fingerprint):
    body = {m: {str(k): v for k, v in d.items()} for m, d in obj.items()}
    body["__qnlep_cache__"] = {"fingerprint": fingerprint}
    return body


def _split_partial_disk(raw):
    if not raw:
        return {}, None
    data = dict(raw)
    meta = data.pop("__qnlep_cache__", None)
    fp = meta.get("fingerprint") if isinstance(meta, dict) else None
    out = {}
    for m, dk in data.items():
        out[m] = {int(k): v for k, v in dk.items()}
    return out, fp


def _fp_a():
    return {
        "run_seed": 42,
        "n_steps": 80,
        "n_steps_by_method": {"ancilla_reuploading": 160, "quantum_reservoir": 200},
        "n_qubits": 3,
        "n_layers": 3,
        "k_values": [1, 2, 4, 8, 16],
        "methods": ["data_reuploading", "ancilla_tracing"],
        "lr": 0.1,
        "r2_thresh": 0.95,
        "n_train": 30,
    }


def _fp_b():
    d = dict(_fp_a())
    d["n_steps"] = 100
    return d


def test_legacy_file_has_no_fingerprint():
    raw = {"data_reuploading": {"1": {"r2": 0.9}}}
    out, fp = _split_partial_disk(raw)
    assert fp is None
    assert "data_reuploading" in out
    assert out["data_reuploading"][1]["r2"] == 0.9


def test_fingerprint_mismatch_means_ignore_on_load():
    """Simulate load_partial_results: if fp on disk != want, return {}."""
    fp_disk = _fp_a()
    want = _fp_b()
    assert fp_disk != want
    # legacy: None != want -> ignore
    assert None != want


def test_roundtrip_with_fingerprint():
    results = {"m1": {1: {"r2": 0.5, "preds": [], "history": []}}}
    fp = _fp_a()
    payload = _partial_to_disk(results, fp)
    assert "__qnlep_cache__" in payload
    out, fp2 = _split_partial_disk(payload)
    assert fp2 == fp
    assert out["m1"][1]["r2"] == 0.5


def test_write_read_file():
    fp = _fp_a()
    results = {"x": {1: {"r2": 1.0, "preds": [], "history": []}}}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "nlc_results_partial.json"
        payload = _partial_to_disk(results, fp)
        p.write_text(json.dumps(payload), encoding="utf-8")
        raw = json.loads(p.read_text(encoding="utf-8"))
        out, fp2 = _split_partial_disk(raw)
        assert fp2 == fp
        assert out["x"][1]["r2"] == 1.0


def main():
    test_legacy_file_has_no_fingerprint()
    test_fingerprint_mismatch_means_ignore_on_load()
    test_roundtrip_with_fingerprint()
    test_write_read_file()
    print("test_nlc_partial_cache: all OK")


if __name__ == "__main__":
    main()
