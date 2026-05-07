"""
TaskKV scorer throughput benchmark (algorithmic-level).

This script measures the CPU-level scoring overhead of the Python
reference implementation to verify that per-step cost is constant
with respect to trajectory length (i.e., does not scale with T).

Measures:
  - Scorer.forward() latency per step (Python, no GPU)
  - Entropy safeguard check overhead
  - EMA intent update cost

The paper's system-level latency table (Table 6) uses a
PagedAttention-style serving prototype with full GPU decode.

Usage:
    python benchmarks/benchmark_throughput.py
    python benchmarks/benchmark_throughput.py --dim 4096 --iterations 500000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from taskkv.policy.entropy_switch import EntropySafeguard
from taskkv.policy.scorer import TaskKVScorer
from taskkv.utils.tensor import Tensor


def run_benchmark(dim: int = 128, iterations: int = 200000):
    scorer = TaskKVScorer(
        hidden_dim=dim,
        lambda_semantic=1.0,
        lambda_noise=1.2,
    )
    safeguard = EntropySafeguard(tau=0.8)

    print("=" * 60)
    print("  TaskKV Scoring Throughput Benchmark")
    print("=" * 60)
    print(f"  Hidden dim : {dim}")
    print(f"  Iterations : {iterations:,}")
    print()

    num_steps = 200
    steps = [Tensor.randn((dim,), seed=i).normalize() for i in range(num_steps)]
    intent = Tensor.randn((dim,), seed=999).normalize()

    # Warmup.
    for _ in range(50):
        scorer.forward(steps[0], intent, 0.5, pivot=0.0)

    # Benchmark: scorer only.
    print("[1/3] Benchmarking scorer.forward() ...")
    t0 = time.perf_counter()
    for i in range(iterations):
        scorer.forward(steps[i % num_steps], intent, 0.5, pivot=0.0)
    t1 = time.perf_counter()
    scorer_dur = t1 - t0
    scorer_tput = iterations / scorer_dur

    # Benchmark: scorer plus entropy check.
    print("[2/3] Benchmarking scorer + entropy safeguard ...")
    scores_buf = [0.0] * 50
    t0 = time.perf_counter()
    for i in range(iterations):
        result = scorer.forward(steps[i % num_steps], intent, 0.5, pivot=0.0)
        scores_buf[i % 50] = result["score"]
        if i % 16 == 0:
            safeguard.should_use_taskkv(scores_buf)
    t1 = time.perf_counter()
    full_dur = t1 - t0
    full_tput = iterations / full_dur

    # Benchmark: EMA intent update.
    print("[3/3] Benchmarking EMA intent update ...")
    alpha = 0.1
    q = Tensor.zeros((dim,))
    t0 = time.perf_counter()
    for i in range(iterations):
        h = steps[i % num_steps]
        q = (h * alpha + q * (1.0 - alpha)).normalize()
    t1 = time.perf_counter()
    ema_dur = t1 - t0
    ema_tput = iterations / ema_dur

    # Report.
    print()
    print("-" * 60)
    print(f"{'Component':<30} {'Throughput':>15} {'Latency':>12}")
    print("-" * 60)
    print(f"{'Scorer only':<30} {scorer_tput:>12,.0f} /s {scorer_dur/iterations*1e6:>9.2f} us")
    print(f"{'Scorer + entropy':<30} {full_tput:>12,.0f} /s {full_dur/iterations*1e6:>9.2f} us")
    print(f"{'EMA intent update':<30} {ema_tput:>12,.0f} /s {ema_dur/iterations*1e6:>9.2f} us")
    print("-" * 60)
    print()
    print("NOTE: These are Python-simulation numbers. See Table 6")
    print("      for full GPU serving latency measurements.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=200000)
    args = parser.parse_args()
    run_benchmark(dim=args.dim, iterations=args.iterations)


if __name__ == "__main__":
    main()
