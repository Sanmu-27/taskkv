# Reproducibility Guide

This document describes how to reproduce the TaskKV experiments and how the
included files map to the paper.

## 1. What can be run locally

Quick verification:

```bash
pip install -e .
python -m taskkv.demo
python benchmarks/verify_reported_results.py
```

This local demo verifies:

- scorer control flow,
- entropy safeguard behavior,
- budgeted memory-manager logic,
- synchronized step-retention bookkeeping on a synthetic trajectory.

The reported-results verifier recomputes Table 1 summaries, paired baseline
deltas, full-cache retention, proxy gaps, and the main AlfWorld ablation gaps
from the released per-seed CSV logs. It verifies the released records and
arithmetic. It does **not** rerun the benchmark environments.

## 2. What the included scripts do

Training pipeline:

- `scripts/collect_trajectories.py`
- `scripts/train_scorer.py`

These scripts document the signed first-order label pipeline and RankNet scorer
training recipe used by TaskKV-Proxy.

Evaluation scripts:

- `scripts/eval_agentbench.py`
- `scripts/eval_alfworld.py`

These scripts run compressed ReAct rollouts with KV-cache eviction on the
respective benchmarks, saving per-seed success rates and computing summary
statistics.

Throughput benchmark:

- `benchmarks/benchmark_throughput.py`
- `benchmarks/verify_reported_results.py`

The throughput benchmark measures Python-side scoring overhead.
The reported-results verifier recomputes all table summaries from the
released per-seed CSV logs on CPU.

## 3. Reported values and raw logs

The artifact includes:

- `results/reported_table_values.csv`
- `results/reported_table_values.json`
- `results/raw_logs/main_results_per_seed.csv`
- `results/raw_logs/ablation_per_seed.csv`
- `results/raw_logs/run_metadata.json`

These files are included for auditability:

- `reported_table_values.*` contain machine-readable copies of selected values
  reported in the paper.
- `raw_logs/*.csv` contain per-seed result records for the released benchmark
  summaries.
- `run_metadata.json` records the protocol, hardware assumptions, and budget
  definitions associated with those summaries.

## 4. Full reproduction setup

To reproduce the agent-success tables end-to-end:

1. **Install environments:** `pip install agentbench alfworld` (public packages).
2. **Obtain backbone weights:** Download Llama-3.1-70B-Instruct under Meta's
   license and place them in the model directory.
3. **Collect trajectories:** `python scripts/collect_trajectories.py --config configs/scorer_llama3_70b.yaml`
4. **Train scorer:** `python scripts/train_scorer.py --config configs/scorer_llama3_70b.yaml`
5. **Run evaluation:** `python scripts/eval_agentbench.py --config configs/eval_agentbench.yaml`

The serving-latency table (Table 6) additionally requires engine-specific
KV-cache hooks for the target inference engine (e.g., vLLM with PagedAttention).

## 5. Scope

This artifact provides an anonymized reference implementation of all
algorithmic components, the complete training and evaluation pipeline,
per-seed raw experiment logs, and a CPU-only verifier. The evaluation
scripts run end-to-end once the external environments and backbone
weights are installed.
