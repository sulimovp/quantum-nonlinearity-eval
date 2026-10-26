# QNLEP — Quantum Non-Linearity Evaluation Protocol (NeurIPS 2026, camera-ready)

Code and data for the paper

> P. Sulimov and C. Lehmann, *A Reproducible Evaluation Protocol for Quantum Non-Linearity in Variational Quantum Models*,
> Advances in Neural Information Processing Systems 39 (NeurIPS 2026), Evaluations and Datasets Track.

**Authors and provenance.** Pavel Sulimov and Claude Lehmann, Zurich University of Applied Sciences (ZHAW), Switzerland. This repository is
the code and data state matching the camera-ready version of the paper. The same code was provided as supplementary material during review.
**Licence:** CC BY 4.0 (see `LICENSE`). **Version:** the git tag `camera-ready` marks the state that matches the published paper.

**Funding.** This work was partially funded by the Institute of Computer Science at Zurich University of Applied Sciences through the project
No. 79838 "QML – Quantum Machine Learning" (https://www.zhaw.ch/en/research/project/79838).

## What the protocol measures

QNLEP compares mechanisms that make variational quantum models non-linear (data re-uploading, ancilla tracing, ancilla re-uploading,
strongly entangling layers, QCNN, quantum reservoir) at a declared width and depth `(n, L)`, with four component scores and a composite:

- **NLC**, non-linear capacity: area under the positive-clipped `R^2(k)` curve for fitting `sin(kx)`, `k` up to 16;
- **BPPI**, barren-plateau proximity: gradient-variance scaling over `n = 2, 4, 6, 8` for a fixed probe loss;
- **NSS**, noise stability: trained models evaluated under a depolarizing-noise ladder;
- **TER**, trainability efficiency on Lorenz one-step prediction;
- **NLPS**, the pre-registered weighted composite, reported alongside the components.

## Setup

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/
```

## Reproduce

The protocol runs as an ordered sequence of notebooks (`03 -> 04 -> 07 -> 08 -> 11 -> 09`, then `06` hybrid baselines and `10` real data).
One command runs them headlessly, in order, from the repository root:

```bash
python scripts/run_neurips_pipeline.py
```

Results are written as JSON under `results/` and figures under `figures/`; each run logs to `pipeline_runs/<UTC timestamp>/`. The global seed
is set with `QNLEP_SEED` (default 42) and the operating point with `QNLEP_N_QUBITS` and `QNLEP_N_LAYERS` (see `src/qnlep_run_paths.py`;
sweeps also set `QNLEP_RESULTS_SUBDIR`); changing seed, width, depth or method list invalidates cached partial runs. At `(n, L) = (3, 3)` the full protocol
takes about two hours on an eight-core laptop CPU.

| Paper element | How to regenerate |
|---------------|-------------------|
| Multi-seed table bodies in `neurips2026/generated/` (seeds 42-46 at `(n, L) = (3, 3)`) | `python scripts/multiseed_paper_pipeline.py` (`--export-only` re-exports from the stored sweeps) |
| (4,4) scaling appendix | `python scripts/run_qnlep_sweep.py` at `(n, L) = (4, 4)`, then `python scripts/paper_export_scaling_table.py` |
| Real-data table and heatmap | `python scripts/run_real_data_all_seeds.py`, then `python scripts/plot_real_data_heatmap.py` |
| BPPI probe-loss ablation | `python scripts/bppi_probe_ablation.py` |
| Protocol vs downstream correlations | `python scripts/downstream_protocol_correlation.py` |
| Weight-sensitivity analysis | `python scripts/weight_sensitivity_analysis.py` |
| Metric intuition figure | `python scripts/generate_metric_intuition_figure.py` |
| Single-method NLC sweep (parallel precompute) | `python scripts/run_nlc_method.py` |

Each script's docstring lists its arguments. The stored per-seed outputs behind every table are in `results/sweeps/n3_L3/` and
`results/sweeps/n4_L4/`.

## Layout

| Path | Contents |
|------|----------|
| `src/` | circuit implementations (`src/circuits/`), protocol metrics (`src/metrics/`), method registry (`qnlep_registry.py`), datasets, run paths |
| `notebooks/` | the protocol stages (outputs cleared; rerun them to regenerate) |
| `scripts/` | headless pipeline, multi-seed sweeps, table export and analyses |
| `tests/` | unit tests |
| `results/` | reference-run JSON, per-seed sweeps, ablation, correlation and weight-sensitivity outputs |
| `figures/` | figures written by the notebooks |
| `neurips2026/` | camera-ready LaTeX source (`main_revision.tex`, `checklist_revision.tex`, generated tables and figures) |

To build the paper: `cd neurips2026 && xelatex main_revision.tex` (run twice).

## Citation

See `CITATION.cff`.
