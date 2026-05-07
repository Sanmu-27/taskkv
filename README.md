# TaskKV

> **Task-Aware KV Cache Compression for LLM Agents via Utility-Driven Step Pruning**
>
> Anonymous Authors -- NeurIPS 2026 Submission

---

## Overview

This repository contains the anonymized reference implementation of **TaskKV**, a step-level KV-cache retention framework for online LLM agents.

## Quick Verification (no GPU required)

```bash
pip install -e .
python -m taskkv.demo
python benchmarks/verify_reported_results.py
```

The demo runs a deterministic local sanity check of the scoring logic, entropy
safeguard, and heap-based memory manager on a synthetic trajectory. The reported
results verifier reads the released per-seed CSV logs, recomputes the main
means/standard deviations, paired deltas against H2O/SnapKV/Strong Heuristic,
full-cache retention, proxy gap, and the main AlfWorld ablation gaps.

These commands are intended for artifact inspection and fast audit of the
released results. They do not rerun AgentBench or AlfWorld.

## Included Components

- **Scorer module** (`taskkv/policy/scorer.py`) -- scoring function matching Eq. 7
- **Memory manager** (`taskkv/core/memory_manager.py`) -- heap-based budgeted retention (Alg. 1)
- **Entropy safeguard** (`taskkv/policy/entropy_switch.py`) -- density-aware fallback (Eq. 8)
- **Online simulator** (`taskkv/simulation.py`) -- streaming inference loop
- **Reported-results verifier** (`benchmarks/verify_reported_results.py`) --
  CPU-only audit over released per-seed logs
- **Training scripts** (`scripts/collect_trajectories.py`, `scripts/train_scorer.py`) -- signed first-order action-utility label generation and RankNet scorer training utilities
- **Evaluation scripts** (`scripts/eval_agentbench.py`, `scripts/eval_alfworld.py`) -- compressed ReAct rollout with KV-cache eviction, per-seed result tracking, and summary statistics
- **Configuration files** (`configs/`) -- hyperparameters matching the paper (Table 8)
- **Reported values** (`results/`) -- machine-readable copies of selected reported table values and provenance notes; these are for auditability, not raw logs

## Repository Structure

```text
taskkv/
|-- core/
|   |-- memory_manager.py      # Budgeted heap retention (Alg. 1)
|   `-- types.py                # Step / TrajectorySegment dataclasses
|-- modeling/
|   `-- layers.py               # Synchronized multi-layer KV cache with eviction
|-- policy/
|   |-- scorer.py               # Scoring function (Eq. 7)
|   |-- entropy_switch.py       # Density-aware safeguard (Eq. 8)
|   `-- baselines.py            # Baseline wrappers (H2O, SnapKV, StreamingLLM, Strong Heuristic, AdaKV)
|-- utils/
|   |-- config.py               # Hyperparameter configuration
|   `-- tensor.py               # Portable tensor utilities
|-- simulation.py               # Online streaming simulator
`-- demo.py                     # Quick verification entry point

scripts/
|-- collect_trajectories.py     # Trajectory collection + signed utility labels
|-- train_scorer.py             # RankNet scorer training with pivot feature
|-- toy_demo_hf.py              # Slow-path HuggingFace end-to-end demo
|-- eval_agentbench.py          # AgentBench benchmark-driver interface
`-- eval_alfworld.py            # AlfWorld benchmark-driver interface

configs/                        # Hyperparameters matching the paper
benchmarks/                     # Scoring throughput measurement
                                # and reported-results verifier
results/                        # Reported values and provenance notes
```

## Citation

```text
Anonymized for review.
```

## Scope of This Artifact

This archive provides a complete reference implementation of TaskKV's
algorithmic components together with the full training and evaluation pipeline.

- **Local verification** — the demo, simulator, and reported-results verifier
  run on CPU without external model weights or environments.
- **Training pipeline** — `collect_trajectories.py` and `train_scorer.py`
  implement the signed-utility label generation and RankNet scorer training.
- **Evaluation pipeline** — `eval_agentbench.py` and `eval_alfworld.py` run
  compressed ReAct rollouts with scorer-driven KV-cache eviction and report
  per-seed success rates.
- **End-to-end demo** — `toy_demo_hf.py` demonstrates the full TaskKV
  eviction loop (scoring, intent tracking, synchronized layer eviction) on a
  live HuggingFace model.
- **Full reproduction** — install the public task environments (`pip install
  agentbench alfworld`), download backbone weights under their licenses, and
  run the included pipeline scripts. See `REPRODUCIBILITY.md` for step-by-step
  instructions.

## License

This code is provided for anonymous review purposes.
