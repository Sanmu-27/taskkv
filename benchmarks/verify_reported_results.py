"""
Fast verifier for the reported TaskKV result logs.

Performs an auditable check over the released per-seed CSV logs:

* recompute mean and sample standard deviation for Table 1,
* compare TaskKV against H2O, SnapKV, Strong Heuristic, and Full Cache,
* estimate per-seed paired deltas where seeds are shared,
* recompute the main AlfWorld ablation gaps.

Run from the artifact root:

    python benchmarks/verify_reported_results.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAIN_LOG = RESULTS / "raw_logs" / "main_results_per_seed.csv"
ABLATION_LOG = RESULTS / "raw_logs" / "ablation_per_seed.csv"
REPORTED = RESULTS / "reported_table_values.csv"


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def sample_std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_main(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["environment"])].append(row)
    return grouped


def group_ablation(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["configuration"]].append(row)
    return grouped


def by_seed(rows: List[Dict[str, str]]) -> Dict[int, float]:
    return {int(row["seed"]): float(row["success_rate"]) for row in rows}


def paired_delta(
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]],
    env: str,
    method_a: str,
    method_b: str,
) -> Tuple[float, float, int]:
    a = by_seed(grouped[(method_a, env)])
    b = by_seed(grouped[(method_b, env)])
    seeds = sorted(set(a).intersection(b))
    deltas = [a[s] - b[s] for s in seeds]
    return mean(deltas), sample_std(deltas), len(deltas)


def load_reported() -> Dict[Tuple[str, str, str], Tuple[float, float]]:
    out = {}
    for row in read_csv(REPORTED):
        if row["table"] != "main_results" or row["metric"] != "SR":
            continue
        method = "TaskKV-Proxy" if row["method"] == "TaskKV-Proxy" else row["method"]
        env = row["setting"]
        std = float(row["std"]) if row["std"] else 0.0
        out[(method, env, row["metric"])] = (float(row["value"]), std)
    return out


def print_main_table(grouped: Dict[Tuple[str, str], List[Dict[str, str]]]) -> None:
    envs = ["AgentBench-OS", "AgentBench-DB", "AgentBench-KG", "AlfWorld"]
    methods = [
        "Full Cache",
        "Window",
        "StreamingLLM",
        "H2O",
        "SnapKV",
        "SqueezeAttn+H2O",
        "Strong Heuristic",
        "AdaKV",
        "Offline SMI Diagnostic",
        "TaskKV-Proxy",
    ]
    reported = load_reported()

    print("Recomputed Table 1 values from per-seed logs")
    print(f"{'method':<24} " + " ".join(f"{env:>18}" for env in envs))
    print("-" * 104)
    for method in methods:
        cells = []
        for env in envs:
            vals = [float(r["success_rate"]) for r in grouped[(method, env)]]
            mu, sd = mean(vals), sample_std(vals)
            rep = reported.get((method, env, "SR"))
            marker = ""
            if rep and (abs(mu - rep[0]) > 0.06 or abs(sd - rep[1]) > 0.06):
                marker = "*"
            cells.append(f"{mu:5.1f}+/-{sd:<4.1f}{marker}")
        print(f"{method:<24} " + " ".join(f"{c:>18}" for c in cells))
    print("* indicates a mismatch above 0.06 against reported_table_values.csv")
    print()


def print_core_deltas(grouped: Dict[Tuple[str, str], List[Dict[str, str]]]) -> None:
    envs = ["AgentBench-OS", "AgentBench-DB", "AgentBench-KG", "AlfWorld"]
    baselines = ["H2O", "SnapKV", "Strong Heuristic"]
    print("TaskKV paired deltas over shared seeds")
    print(f"{'comparison':<34} {'mean delta':>12} {'std':>8} {'n':>4}")
    print("-" * 64)
    for env in envs:
        for base in baselines:
            delta, sd, n = paired_delta(grouped, env, "TaskKV-Proxy", base)
            print(f"{env + ' vs ' + base:<34} {delta:>11.2f} {sd:>8.2f} {n:>4}")
    print()

    print("TaskKV retention relative to Full Cache")
    print(f"{'environment':<18} {'TaskKV':>8} {'Full':>8} {'retention':>10}")
    print("-" * 50)
    for env in envs:
        taskkv = mean(float(r["success_rate"]) for r in grouped[("TaskKV-Proxy", env)])
        full = mean(float(r["success_rate"]) for r in grouped[("Full Cache", env)])
        print(f"{env:<18} {taskkv:>8.1f} {full:>8.1f} {100.0 * taskkv / full:>9.1f}%")
    print()

    print("Proxy gap relative to Offline SMI Diagnostic")
    print(f"{'environment':<18} {'TaskKV':>8} {'SMI':>8} {'gap':>8}")
    print("-" * 46)
    for env in envs:
        taskkv = mean(float(r["success_rate"]) for r in grouped[("TaskKV-Proxy", env)])
        smi = mean(float(r["success_rate"]) for r in grouped[("Offline SMI Diagnostic", env)])
        print(f"{env:<18} {taskkv:>8.1f} {smi:>8.1f} {taskkv - smi:>8.1f}")
    print()


def print_ablation(ablation_rows: List[Dict[str, str]]) -> None:
    grouped = group_ablation(ablation_rows)
    full = mean(float(r["success_rate"]) for r in grouped["Full TaskKV"])
    print("Recomputed AlfWorld ablation gaps")
    print(f"{'configuration':<34} {'mean':>8} {'std':>8} {'gap':>8}")
    print("-" * 64)
    for name in [
        "Full TaskKV",
        "w/o Denoising (lambda2=0)",
        "w/o Pivot Feature (Psi=0)",
        "w/o Intent EMA (alpha=1.0)",
        "w/o Semantic (lambda1=0)",
        "Unsigned Saliency Labels (|grad| only)",
        "Fixed Stride (S=64)",
        "Fixed Stride (S=32)",
        "Budget=10%",
        "Budget=20%",
        "Budget=30%",
        "Budget=50%",
    ]:
        if name not in grouped:
            continue
        vals = [float(r["success_rate"]) for r in grouped[name]]
        mu, sd = mean(vals), sample_std(vals)
        print(f"{name:<34} {mu:>8.1f} {sd:>8.1f} {mu - full:>8.1f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    main_rows = read_csv(MAIN_LOG)
    ablation_rows = read_csv(ABLATION_LOG)
    grouped = group_main(main_rows)

    print("TaskKV reported-results verifier")
    print(f"main log     : {MAIN_LOG.relative_to(ROOT)}")
    print(f"ablation log : {ABLATION_LOG.relative_to(ROOT)}")
    print()
    print_main_table(grouped)
    print_core_deltas(grouped)
    print_ablation(ablation_rows)
    print("Verification complete. All summaries recomputed from per-seed raw logs.")


if __name__ == "__main__":
    main()
