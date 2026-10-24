"""Tests for sweep / results path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_protocol_results_dir_default(tmp_path, monkeypatch):
    """Explicit results root (default is repo ``results/``, not cwd-relative)."""
    res = tmp_path / "results"
    monkeypatch.setenv("QNLEP_PROTOCOL_RESULTS", str(res))
    monkeypatch.delenv("QNLEP_RESULTS_SUBDIR", raising=False)
    from importlib import reload
    import src.qnlep_run_paths as m

    reload(m)
    p = m.protocol_results_dir()
    assert p.resolve() == res.resolve()
    assert p.is_dir()


def test_protocol_results_dir_with_subdir(tmp_path, monkeypatch):
    res = tmp_path / "results"
    monkeypatch.setenv("QNLEP_PROTOCOL_RESULTS", str(res))
    monkeypatch.setenv("QNLEP_RESULTS_SUBDIR", "sweeps/n3_L3/seed_99")
    from importlib import reload
    import src.qnlep_run_paths as m

    reload(m)
    p = m.protocol_results_dir()
    assert "sweeps" in str(p)
    assert p.is_dir()


def test_protocol_n_qubits_layers(monkeypatch):
    monkeypatch.delenv("QNLEP_N_QUBITS", raising=False)
    monkeypatch.delenv("QNLEP_N_LAYERS", raising=False)
    from importlib import reload
    import src.qnlep_run_paths as m

    reload(m)
    assert m.protocol_n_qubits(3) == 3
    assert m.protocol_n_layers(2) == 2
    monkeypatch.setenv("QNLEP_N_QUBITS", "4")
    monkeypatch.setenv("QNLEP_N_LAYERS", "5")
    reload(m)
    assert m.protocol_n_qubits(3) == 4
    assert m.protocol_n_layers(2) == 5
