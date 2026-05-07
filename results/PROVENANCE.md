# Reported Results Provenance

This directory contains machine-readable copies of selected numbers reported in
the paper, per-seed raw experiment logs, and scope notes for reviewers.

## Directory Structure

| File | Description |
|---|---|
| `reported_table_values.csv` | Aggregated values (mean ± std) used in the paper |
| `reported_table_values.json` | Same values in JSON for programmatic access |
| `raw_logs/main_results_per_seed.csv` | Per-seed raw logs for Table 1 (10 methods × 4 envs × 5 seeds) |
| `raw_logs/ablation_per_seed.csv` | Per-seed raw logs for Table 2 ablation study |
| `raw_logs/run_metadata.json` | Full experiment configuration, protocol, and data hygiene |

## Raw Log Format

Each row in the per-seed CSVs records one (method, environment, seed) run:

- `success_rate` — the authoritative per-seed success rate (%)
- `notes` — domain status (training-matched vs. diagnostic transfer, etc.)

Episode counts per environment are documented in `raw_logs/run_metadata.json`.
The reported mean ± std in the paper are computed from the `success_rate` column.

## Verification

Run the CPU-only verifier to recompute all table summaries from the per-seed logs:

```bash
python benchmarks/verify_reported_results.py
```

This recomputes Table 1 means/stds, paired baseline deltas, full-cache retention
percentages, proxy gaps vs. the offline SMI diagnostic, and the main AlfWorld
ablation gaps. See `REPRODUCIBILITY.md` in the repository root for full setup
instructions.

## Coverage of Raw Logs vs. Aggregate Values

| Table / Result | Raw per-seed logs included? | Source |
|---|---|---|
| Table 1 (main results) | ✅ `raw_logs/main_results_per_seed.csv` | 10 methods × 4 envs × 5 seeds |
| Table 2 (ablation) | ✅ `raw_logs/ablation_per_seed.csv` | 12 configs × 5 seeds |
| Table 6 (throughput) | ❌ aggregate only | Serving-stack profiling requires engine-specific hooks; values in `reported_table_values.csv` |
| Table 4/5 (generalization) | ❌ aggregate only | Transfer experiments; aggregate values in `reported_table_values.csv` |
| Proxy validation (r_pb, τ) | ❌ aggregate only | Offline diagnostic; aggregate values in `reported_table_values.csv` |
| Tau sensitivity sweep | ❌ aggregate only | Appendix diagnostic; aggregate values in `reported_table_values.csv` |

Per-seed raw logs for the secondary tables will be released with the
de-anonymized code upon acceptance. The current anonymous submission
prioritizes full auditability of the central claims (Tables 1 and 2).

