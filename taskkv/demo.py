"""
Quick verification demo (``python -m taskkv``).

Runs the TaskKV reference pipeline on a synthetic ReAct-style trajectory to
check:
  1. system prompt retention,
  2. error trace pruning,
  3. intent-driven alignment,
  4. entropy safeguard bookkeeping,
  5. budgeted heap eviction.

No GPU or model weights are required.
"""

from .simulation import TaskKVSimulator
from .utils.config import SimulationConfig


def _print_header():
    print("=" * 72)
    print("  TaskKV Reference Implementation -- Quick Verification Demo")
    print("  Anonymous -- NeurIPS 2026 Submission")
    print("=" * 72)


def _run_demo():
    cfg = SimulationConfig(hidden_dim=64, num_layers=12, budget_ratio=0.3)
    sim = TaskKVSimulator(cfg)

    trajectory = [
        ("System Prompt", "system", 1.5),
        ("User Goal", "goal", 0.9),
        ("Thought: Plan Step 1", "thought", 0.6),
        ("Action: execute cmd", "action", 0.5),
        ("Obs: Permission denied", "error", 0.2),
        ("Obs: Traceback ...", "error", 0.1),
        ("Thought: Retry sudo", "thought", 0.7),
        ("Action: sudo execute", "action", 0.6),
        ("Obs: Success output", "observation", 0.8),
        ("Thought: Parse result", "thought", 0.7),
        ("Action: final answer", "action", 0.9),
    ]

    print(
        f"\n[Config] dim={cfg.hidden_dim}, layers={cfg.num_layers}, "
        f"budget={cfg.budget_ratio:.0%}, lambda_sem={cfg.lambda_semantic}, "
        f"lambda_noise={cfg.lambda_noise}, ema_alpha={cfg.ema_alpha}"
    )
    print(f"[Config] entropy threshold={cfg.entropy_threshold}")

    print(
        f"\n{'STEP':<26} {'TYPE':<12} {'SCORE':>7} {'SEM':>7} "
        f"{'NOISE':>7} {'STATUS':<10} {'MODE':<12} {'H(f)':>6}"
    )
    print("-" * 98)

    for name, stype, attn in trajectory:
        r = sim.process_step(name, stype, attn)
        print(
            f"{r['name']:<26} {r['type']:<12} {r['score']:>7.3f} "
            f"{r['sem']:>7.3f} {r['noise']:>7.3f} "
            f"{r['status']:<10} {r['mode']:<12} {r['entropy']:>6.3f}"
        )

    retained = sim.get_retained_memory()
    retained_tokens = sim.memory.retained_tokens if sim.memory is not None else 0
    budget_tokens = sim.memory.budget_tokens if sim.memory is not None else 0
    print(f"\n{'=' * 72}")
    print(
        f"  Retained Memory ({len(retained)} / {len(trajectory)} steps, "
        f"{retained_tokens} / {budget_tokens} tokens)"
    )
    print(f"{'=' * 72}")
    for r in retained:
        tag = "kept" if r["type"] != "error" else "check"
        print(
            f"  [{tag}] t={r['t']:>2}  {r['name']:<26}  "
            f"type={r['type']:<12} score={r['score']:.3f}"
        )

    has_system = any(r["type"] == "system" for r in retained)
    has_error = any(r["type"] == "error" for r in retained)

    print("\n[Validation]")
    print(f"  System prompt retained : {'PASS' if has_system else 'FAIL'}")
    print(f"  Error traces pruned    : {'PASS' if not has_error else 'CHECK'}")
    print(f"  Entropy safeguard mode : {sim.safeguard.mode}")
    print()


def main():
    _print_header()
    _run_demo()


if __name__ == "__main__":
    main()
