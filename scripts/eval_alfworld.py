"""
AlfWorld evaluation script for TaskKV and baselines.

Runs compressed ReAct rollouts on AlfWorld with the specified KV-cache
compression policy. Results are saved per-seed and aggregated into
mean +/- std summaries matching the paper's Table 1.

Usage:
    python scripts/eval_alfworld.py \
        --model_path /path/to/Llama-3.1-70B-Instruct \
        --scorer_path checkpoints/scorer/best.pt \
        --method taskkv \
        --budget_ratio 0.20 \
        --seeds 1000-1004
"""

import argparse
import csv
import json
import math
import os
import sys
import time


def parse_seed_range(text: str):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        print("[ERROR] PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def sample_std(vals):
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))


def verify_dependencies(model_path: str, method: str, scorer_path: str = None):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at '{model_path}'.")
        sys.exit(1)

    try:
        import alfworld  # noqa: F401
        import alfworld.agents  # noqa: F401
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}")
        sys.exit(1)

    if method == "taskkv" and (scorer_path is None or not os.path.exists(scorer_path)):
        print("[ERROR] Scorer checkpoint required for TaskKV method.")
        sys.exit(1)


def load_model(model_path: str, config: dict):
    """Load backbone LLM with KV-cache support."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    model.eval()
    return model, tokenizer


def setup_compression_policy(method: str, config: dict, scorer_path: str = None):
    """Build the KV-cache compression policy."""
    if method == "taskkv":
        from taskkv.core.memory_manager import MemoryManager
        from taskkv.policy.entropy_switch import EntropySafeguard
        from taskkv.policy.scorer import TaskKVScorer

        hidden_dim = config.get("hidden_dim", 8192)
        scorer = TaskKVScorer(
            hidden_dim=hidden_dim,
            lambda_semantic=config.get("lambda_semantic", 1.0),
            lambda_noise=config.get("lambda_noise", 1.2),
        )
        scorer._mlp.load_weights(scorer_path)
        return {
            "type": "taskkv",
            "scorer": scorer,
            "memory_manager_cls": MemoryManager,
            "safeguard": EntropySafeguard(tau=config.get("entropy_threshold", 0.8)),
        }

    if method == "h2o":
        from taskkv.policy.baselines import H2OPolicy
        return {"type": "h2o", "policy": H2OPolicy(protect_ratio=config.get("h2o_protect_ratio", 0.1))}

    if method == "streamingllm":
        from taskkv.policy.baselines import StreamingLLMPolicy
        return {"type": "streamingllm", "policy": StreamingLLMPolicy(sink_tokens=config.get("sink_tokens", 4))}

    if method == "snapkv":
        from taskkv.policy.baselines import SnapKVPolicy
        return {"type": "snapkv", "policy": SnapKVPolicy(
            window_size=config.get("snapkv_window_size", 64),
            sink_tokens=config.get("sink_tokens", 4),
        )}

    if method == "strong_heuristic":
        from taskkv.policy.baselines import StrongHeuristicPolicy
        return {"type": "strong_heuristic", "policy": StrongHeuristicPolicy(
            protect_ratio=config.get("heuristic_protect_ratio", 0.1),
            recent_ratio=config.get("heuristic_recent_ratio", 0.3),
        )}

    if method == "adakv":
        from taskkv.policy.baselines import AdaKVPolicy
        return {"type": "adakv", "policy": AdaKVPolicy(
            protect_ratio=config.get("adakv_protect_ratio", 0.1),
            recent_ratio=config.get("adakv_recent_ratio", 0.2),
        )}

    if method == "full":
        return {"type": "full"}

    print(f"[ERROR] Unknown method '{method}'")
    sys.exit(1)


def run_episode(model, tokenizer, env, seed, policy, budget_ratio, max_steps=200):
    """Run one AlfWorld ReAct episode with KV-cache compression."""
    import torch

    obs, info = env.reset()
    admissible = info.get("admissible_commands", [])

    system_prompt = (
        "You are an agent in a household environment. "
        "Use the ReAct format:\nThought: <reasoning>\nAction: <command>\n"
    )
    history = [system_prompt, f"Observation: {obs}"]

    past_key_values = None
    step_boundaries = []
    success = False

    for step_idx in range(max_steps):
        prompt_text = "\n".join(history) + "\nThought:"
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

        past_key_values = outputs.past_key_values
        kv_len = past_key_values[0][0].shape[2]

        step_boundaries.append({
            "start": kv_len - input_ids.shape[1],
            "end": kv_len,
            "step_idx": step_idx,
        })

        gen_ids = model.generate(
            input_ids,
            past_key_values=past_key_values,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
        )
        response = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

        action_text = None
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Action:"):
                action_text = stripped[len("Action:"):].strip()
                break

        if action_text is None:
            history.append(f"Thought: {response}")
            continue

        history.append(f"Thought: {response.split('Action:')[0].strip()}")
        history.append(f"Action: {action_text}")

        try:
            obs, reward, done, info = env.step(action_text)
        except Exception as exc:
            obs, reward, done = f"Error: {exc}", 0, False

        history.append(f"Observation: {obs}")

        # Apply KV-cache compression
        budget_tokens = int(budget_ratio * kv_len)
        if policy["type"] == "taskkv" and kv_len > budget_tokens:
            past_key_values = _apply_taskkv_eviction(
                past_key_values, outputs.hidden_states, policy, step_boundaries, budget_tokens,
            )
        elif policy["type"] != "full" and kv_len > budget_tokens:
            past_key_values = _slice_kv(past_key_values, budget_tokens)

        if done:
            success = reward > 0
            break

    return {"seed": seed, "success": success, "num_steps": step_idx + 1}


def _apply_taskkv_eviction(past_key_values, hidden_states, policy, step_boundaries, budget_tokens):
    """Apply TaskKV scorer-based synchronized eviction."""
    import torch

    kv_len = past_key_values[0][0].shape[2]
    if kv_len <= budget_tokens:
        return past_key_values

    last_hidden = hidden_states[-1][0]
    scorer = policy["scorer"]
    safeguard = policy["safeguard"]

    scores = []
    for boundary in step_boundaries:
        start = min(boundary["start"], last_hidden.shape[0])
        end = min(boundary["end"], last_hidden.shape[0])
        if start == end:
            scores.append(0.0)
            continue
        h_i = last_hidden[start:end].mean(dim=0)
        h_norm = h_i / (h_i.norm() + 1e-8)
        intent = last_hidden[-1] / (last_hidden[-1].norm() + 1e-8)
        from taskkv.utils.tensor import Tensor
        result = scorer.forward(
            Tensor(h_norm.cpu().tolist(), tuple(h_norm.shape)),
            Tensor(intent.cpu().tolist(), tuple(intent.shape)),
            attn_norm=0.5, pivot=0.0,
        )
        scores.append(result["score"])

    if not safeguard.should_use_taskkv(scores):
        return _slice_kv(past_key_values, budget_tokens)

    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    positions_to_remove = []
    tokens_removed = 0
    target = kv_len - budget_tokens

    for idx, _ in indexed:
        if tokens_removed >= target:
            break
        boundary = step_boundaries[idx]
        positions_to_remove.extend(range(boundary["start"], boundary["end"]))
        tokens_removed += boundary["end"] - boundary["start"]

    if positions_to_remove:
        keep_mask = torch.ones(kv_len, dtype=torch.bool, device=past_key_values[0][0].device)
        for pos in positions_to_remove:
            if 0 <= pos < kv_len:
                keep_mask[pos] = False
        new_past = []
        for k, v in past_key_values:
            new_past.append((k[:, :, keep_mask, :], v[:, :, keep_mask, :]))
        return tuple(new_past)

    return past_key_values


def _slice_kv(past_key_values, keep_len):
    new_past = []
    for k, v in past_key_values:
        new_past.append((k[:, :, -keep_len:, :], v[:, :, -keep_len:, :]))
    return tuple(new_past)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TaskKV on AlfWorld")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scorer_path", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/eval_alfworld.yaml")
    parser.add_argument(
        "--method", type=str, default="taskkv",
        choices=["taskkv", "h2o", "streamingllm", "snapkv", "strong_heuristic", "adakv", "full"],
    )
    parser.add_argument("--budget_ratio", type=float, default=0.2)
    parser.add_argument("--seeds", type=str, default="1000-1004")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results/alfworld")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = parse_seed_range(args.seeds)
    verify_dependencies(args.model_path, args.method, args.scorer_path)
    policy = setup_compression_policy(args.method, config, args.scorer_path)
    model, tokenizer = load_model(args.model_path, config)

    print(f"[INFO] Method: {args.method}")
    print(f"[INFO] Budget: {args.budget_ratio:.0%}")
    print(f"[INFO] Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")

    import alfworld
    import alfworld.agents.environment as alfworld_env
    env = alfworld_env.AlfWorldEnv()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for seed in seeds:
        env.seed(seed)
        t0 = time.time()
        result = run_episode(model, tokenizer, env, seed, policy, args.budget_ratio, args.max_steps)
        elapsed = time.time() - t0
        result["method"] = args.method
        result["budget"] = args.budget_ratio
        result["elapsed_s"] = round(elapsed, 1)
        all_results.append(result)

        status = "PASS" if result["success"] else "FAIL"
        print(f"  [{status}] seed={seed} steps={result['num_steps']} ({elapsed:.1f}s)")

    # Save per-episode results
    out_path = os.path.join(args.output_dir, f"{args.method}_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "budget", "success", "num_steps", "elapsed_s"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[INFO] Per-episode results saved -> {out_path}")

    # Summary
    rates = [100.0 * r["success"] for r in all_results]
    mu = sum(rates) / len(rates)
    sd = sample_std(rates)
    print(f"\n{'='*50}")
    print(f"  AlfWorld {args.method} @ {args.budget_ratio:.0%}: {mu:.1f} +/- {sd:.1f} ({len(rates)} seeds)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
