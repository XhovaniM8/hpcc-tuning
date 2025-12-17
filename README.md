# HPCC++ Parameter Tuning for AI Datacenter Congestion Control

This repository contains the code and artifacts for **HPCC++ Parameter Tuning for AI Datacenter Congestion Control: A Control-Theoretic Perspective with Systematic Experimental Validation** (NYU ECE-GY 6383).

It provides:
- A tunable HPCC++ implementation in `csg-htsim`-style simulation code
- Repeatable parameter sweeps over `(alpha, beta)` grouped by `zeta = 2*beta/alpha`
- Mechanism ablations (disable multiplicative clamp / cwnd bound)
- Plotting and CSV outputs for mean/p99/max flow completion time (FCT)

## Repo Layout

- `simulation/`  
  HPCC++ simulator build and experiment runner scripts.
  - `simulation/run_experiments.sh`: main entrypoint for running experiments
  - `simulation/traffic/`: traffic matrices (heavy, incast, etc.)
  - `simulation/hpcc/`: HPCC++ implementation

- `analysis/`  
  Parsing + analysis drivers.
  - `analysis/run_hpcc_experiments.py`: orchestrates sweeps/ablations
  - `analysis/plot_hpcc_results.py`: generates plots into `results/figures/`

- `results/`  
  Generated outputs.
  - `results/data/`: CSV/JSONL summaries
  - `results/figures/`: PDFs used in the talk/paper

- `docs/`  
  Writeups and references.
  - `docs/HPCC_TUNING_GUIDE.md`
  - `docs/PARAMETER_REFERENCE.md`
  - `docs/paper/term_project_paper.tex`

## Requirements

Python dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
