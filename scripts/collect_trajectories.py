"""
Trajectory collection with signed first-order action-utility labels.

This script documents the label-generation path used for TaskKV-Proxy. It
rolls out ReAct-style episodes, records semantic steps, and can compute a
signed first-order utility proxy for each step:

    U_proxy(s_i) = - < grad_{h_i} L_CLM(A*), h_i - h_mask >

Positive labels predict that keeping a step lowers the ground-truth action
loss; negative labels predict that the step is harmful for that action.

Usage:
    python scripts/collect_trajectories.py \
        --model_path <path-to-backbone-model> \
        --env agentbench-os \
        --seeds 0-499 \
        --output_dir data/trajectories/os_train \
        --compute_signed_utility

Prerequisites for full label generation:
    - model weights downloaded to --model_path
    - public AgentBench / AlfWorld environments installed
    - torch and transformers installed
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_seed_range(text: str):
    """Parse a seed specification such as '0-499' or '1,7,9'."""
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def load_model(model_path: str):
    """Load a backbone LLM via HuggingFace Transformers."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install transformers")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at '{model_path}'.")
        print("        Download backbone weights independently and pass --model_path.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def connect_environment(env_name: str):
    """Connect to a public agent evaluation environment."""
    env_registry = {
        "agentbench-os": "agentbench.environments.os",
        "agentbench-db": "agentbench.environments.db",
        "agentbench-kg": "agentbench.environments.kg",
        "alfworld": "alfworld.agents.environment",
    }
    module_name = env_registry.get(env_name)
    if module_name is None:
        print(f"[ERROR] Unknown environment '{env_name}'.")
        print(f"        Available: {list(env_registry.keys())}")
        sys.exit(1)

    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        print(f"[ERROR] Environment package not found for '{env_name}'.")
        print("        Install the upstream benchmark package first.")
        sys.exit(1)


def _step_token_spans(tokenizer, trajectory):
    spans = []
    offset = 0
    for step in trajectory:
        ids = tokenizer(step["text"], add_special_tokens=False).input_ids
        span_len = len(ids)
        spans.append((offset, offset + span_len))
        offset += span_len
    return spans


def _pivot_feature(step):
    text = step.get("text", "").lower()
    stype = step.get("type", "")
    pivot_words = ("retry", "instead", "switch", "new plan", "use find", "need sudo")
    return 1.0 if stype == "goal" or any(w in text for w in pivot_words) else 0.0


def compute_signed_action_utility(model, tokenizer, trajectory, target_action):
    """
    Compute signed first-order action-utility labels for semantic steps.

    The reference implementation uses final-layer hidden states as the
    differentiable centroid path because HuggingFace key-cache tensors are not
    consistently retained as graph leaves across model families. The paper's
    serving path uses pooled key states; both produce the same label type:
    a signed first-order estimate of action-loss change under local masking.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    full_text = "\n".join(step["text"] for step in trajectory)
    action_text = target_action
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    step_spans = _step_token_spans(tokenizer, trajectory)

    model.zero_grad(set_to_none=True)
    outputs = model(full_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    hidden.retain_grad()

    action_ids = tokenizer(
        action_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    action_len = min(action_ids.shape[1], full_ids.shape[1] - 1)
    action_start = full_ids.shape[1] - action_len

    logits = outputs.logits
    shift_logits = logits[:, action_start - 1 : -1, :]
    shift_labels = full_ids[:, action_start:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    loss.backward()

    grad = hidden.grad
    if grad is None:
        return [0.0] * len(trajectory)

    utilities = []
    for start, end in step_spans:
        start = max(0, min(start, hidden.shape[1]))
        end = max(start, min(end, hidden.shape[1]))
        if start == end:
            utilities.append(0.0)
            continue
        h_i = hidden[0, start:end, :].detach().mean(dim=0)
        g_i = grad[0, start:end, :].detach().mean(dim=0)
        h_mask = torch.zeros_like(h_i)
        utility = -torch.dot(g_i, h_i - h_mask).item()
        utilities.append(float(utility))

    scale = max(max(abs(u) for u in utilities), 1e-8)
    return [u / scale for u in utilities]


def parse_react_steps(text: str) -> list:
    """Split a ReAct-format block into semantic steps."""
    import re

    pattern = re.compile(
        r"(Thought:.*?)(?=Thought:|Action:|Observation:|$)|"
        r"(Action:.*?)(?=Thought:|Action:|Observation:|$)|"
        r"(Observation:.*?)(?=Thought:|Action:|Observation:|$)",
        re.DOTALL,
    )
    steps = []
    for match in pattern.finditer(text):
        raw = match.group(0).strip()
        if not raw:
            continue
        if raw.startswith("Thought:"):
            stype = "thought"
        elif raw.startswith("Action:"):
            stype = "action"
        else:
            stype = "observation"
        steps.append({"text": raw, "type": stype, "pivot": _pivot_feature({"text": raw, "type": stype})})
    return steps


def run_episode(model, tokenizer, env, seed: int, compute_signed_utility: bool):
    """
    Run one full-cache ReAct episode and optionally compute signed labels.

    Collects full-cache training trajectories and signed labels for the scorer.
    Compressed evaluation with trained policies is handled by eval_agentbench.py
    and eval_alfworld.py.
    """
    import torch

    if hasattr(env, "create_task"):
        task = env.create_task(seed)
        obs = task.reset()
        env_step_fn = task.step
    else:
        obs = env.reset(seed)
        env_step_fn = env.step

    system_prompt = (
        "You are an autonomous agent. Use the ReAct format:\n"
        "Thought: <reasoning>\nAction: <command>\n"
        "You will receive Observation: <result> after each action.\n"
    )
    history = [{"text": system_prompt, "type": "system", "pivot": 0.0}]
    history.append({"text": f"Observation: {obs}", "type": "observation", "pivot": 0.0})

    success = False
    for _ in range(200):
        prompt = "\n".join(step["text"] for step in history) + "\nThought:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(next(model.parameters()).device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        react_steps = parse_react_steps("Thought:" + generated)
        if not react_steps:
            break
        history.extend(react_steps)

        action_text = None
        for step in react_steps:
            if step["type"] == "action":
                action_text = step["text"].replace("Action:", "").strip()
                break
        if action_text is None:
            break

        try:
            obs, reward, done = env_step_fn(action_text)
        except Exception as exc:
            obs, reward, done = f"Error: {exc}", 0, False

        history.append({"text": f"Observation: {obs}", "type": "observation", "pivot": 0.0})
        if done:
            success = reward > 0
            break

    labels = None
    if compute_signed_utility:
        target_action = None
        for step in reversed(history):
            if step["type"] == "action":
                target_action = step["text"]
                break
        if target_action is not None:
            labels = compute_signed_action_utility(model, tokenizer, history, target_action)

    for step in history:
        step.setdefault("pivot", _pivot_feature(step))

    return {
        "seed": seed,
        "success": success,
        "num_steps": len(history),
        "steps": history,
        "signed_utility": labels,
        "label_type": "signed_first_order_action_utility" if labels is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect TaskKV trajectories")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["agentbench-os", "agentbench-db", "agentbench-kg", "alfworld"],
    )
    parser.add_argument("--seeds", type=str, default="0-499")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--compute_signed_utility", action="store_true")
    args = parser.parse_args()

    compute_labels = args.compute_signed_utility

    seeds = parse_seed_range(args.seeds)
    print(f"[INFO] Environment: {args.env}")
    print(f"[INFO] Episodes: {len(seeds)}")
    print(f"[INFO] Signed utility labels: {compute_labels}")

    model, tokenizer = load_model(args.model_path)
    env = connect_environment(args.env)
    os.makedirs(args.output_dir, exist_ok=True)

    for seed in seeds:
        result = run_episode(model, tokenizer, env, seed, compute_labels)
        out_path = os.path.join(args.output_dir, f"trajectory_{seed:05d}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
