"""
Advanced verification example.

Demonstrates the full TaskKV pipeline on a longer synthetic trajectory
that includes multiple error-recovery cycles and strategy pivots.

Usage:
    python examples/run_advanced.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from taskkv.simulation import TaskKVSimulator
from taskkv.utils.config import SimulationConfig


def run_advanced_verification():
    cfg = SimulationConfig(
        hidden_dim=64,
        num_layers=12,
        budget_ratio=0.4,
    )
    sim = TaskKVSimulator(cfg)

    trajectory = [
        ("System Prompt", "system", 1.5),
        ("User Goal: find file", "goal", 0.9),
        ("Thought: try ls", "thought", 0.6),
        ("Action: ls /home", "action", 0.5),
        ("Obs: file not found", "error", 0.2),
        ("Obs: stack trace", "error", 0.1),
        ("Thought: use find cmd", "thought", 0.7),
        ("Action: find / -name x", "action", 0.6),
        ("Obs: /var/data/x.txt", "observation", 0.8),
        ("Thought: read file", "thought", 0.5),
        ("Action: cat /var/x.txt", "action", 0.4),
        ("Obs: permission denied", "error", 0.2),
        ("Obs: errno 13", "error", 0.1),
        ("Thought: need sudo", "thought", 0.7),
        ("Action: sudo cat file", "action", 0.6),
        ("Obs: file contents", "observation", 0.9),
        ("Thought: parse output", "thought", 0.7),
        ("Action: submit answer", "action", 0.9),
    ]

    print("=" * 80)
    print("  TaskKV Advanced Verification -- Multi-Phase Trajectory")
    print("=" * 80)
    print(
        f"\n  Config: dim={cfg.hidden_dim}, budget={cfg.budget_ratio:.0%}, "
        f"lambda_sem={cfg.lambda_semantic}, lambda_noise={cfg.lambda_noise}"
    )

    print(
        f"\n{'STEP':<26} {'TYPE':<12} {'SCORE':>7} {'SEM':>7} "
        f"{'NOISE':>7} {'STATUS':<10} {'MODE'}"
    )
    print("-" * 90)

    for name, stype, attn in trajectory:
        result = sim.process_step(name, stype, attn)
        print(
            f"{result['name']:<26} {result['type']:<12} {result['score']:>7.3f} "
            f"{result['sem']:>7.3f} {result['noise']:>7.3f} "
            f"{result['status']:<10} {result['mode']}"
        )

    retained = sim.get_retained_memory()
    print(f"\n{'=' * 80}")
    print(f"  Retained: {len(retained)} / {len(trajectory)} steps")
    print(f"{'=' * 80}")
    for step in retained:
        print(
            f"  t={step['t']:>2}  {step['name']:<26}  type={step['type']:<12} "
            f"score={step['score']:.3f}"
        )

    retained_types = [step["type"] for step in retained]
    retained_names = [step["name"] for step in retained]
    print("\n[Checks]")
    print(f"  System prompt retained : {'PASS' if 'system' in retained_types else 'FAIL'}")
    print(f"  Error traces pruned    : {'PASS' if 'error' not in retained_types else 'FAIL'}")
    print(
        "  Goal or solution kept  : "
        f"{'PASS' if ('goal' in retained_types or 'Obs: file contents' in retained_names) else 'FAIL'}"
    )


if __name__ == "__main__":
    run_advanced_verification()
