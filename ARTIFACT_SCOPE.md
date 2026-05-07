# Artifact Scope

This artifact is an anonymized reference implementation of the TaskKV
algorithmic components.

## What can be checked locally

- scoring-function control flow,
- entropy safeguard behavior,
- heap-based budgeted retention,
- synthetic online streaming simulation,
- signed-utility label generation and RankNet training scripts,
- evaluation scripts and configuration schema.

## Implementation note

The paper's reported experiments use pooled final-layer key states for step
centroids and signed first-order labels. The reference label script uses
final-layer hidden-state attribution as the portable equivalent, which
produces the same signed label type across HuggingFace model families.

Run:

```bash
pip install -e .
python -m taskkv
```

## Full benchmark reproduction

To reproduce the benchmark tables end-to-end, install the publicly available
external dependencies and run the included pipeline scripts:

1. `pip install agentbench alfworld` — public task environments.
2. Download backbone weights (Llama-3.1-70B-Instruct) under Meta's license.
3. `python scripts/collect_trajectories.py` — collect trajectories with signed labels.
4. `python scripts/train_scorer.py` — train the scorer MLP via RankNet.
5. `python scripts/eval_agentbench.py` / `eval_alfworld.py` — run evaluation.

The serving-latency table (Table 6) additionally requires engine-specific
KV-cache hooks (e.g., vLLM PagedAttention integration). See `REPRODUCIBILITY.md`
for detailed setup instructions.
