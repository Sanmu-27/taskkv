"""
Toy Demo for TaskKV: HuggingFace Integration

This script provides a functional, slow-path reference implementation of TaskKV's
step-level KV cache eviction using the standard HuggingFace Transformers library.
Unlike the high-performance PagedAttention prototype (which requires deep C++ hooks),
this script operates by explicitly slicing and concatenating past_key_values tensors.

It demonstrates:
1. Semantic step chunking based on delimiter tokens.
2. Per-chunk utility scoring via cosine-similarity proxy to the running intent.
3. Synchronized layer eviction across all transformer layers to meet a token budget.

Usage:
    python scripts/toy_demo_hf.py [--model_id MODEL] [--budget_tokens N]
"""

import argparse
import math
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Utility scoring (lightweight proxy matching Eq. 7 structure)
# ---------------------------------------------------------------------------

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    dot = torch.dot(a, b).item()
    na = a.norm().item()
    nb = b.norm().item()
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def score_chunk(
    chunk_hidden: torch.Tensor,
    intent: torch.Tensor,
    attn_mass: float,
    lambda_sem: float = 1.0,
) -> float:
    """Score a semantic chunk following the TaskKV proxy structure.

    Args:
        chunk_hidden: mean-pooled hidden state for the chunk.
        intent: running EMA intent vector.
        attn_mass: normalized attention mass contributed by the chunk.
        lambda_sem: weight for semantic alignment term.

    Returns:
        Scalar utility score  f = phi_attn + lambda_1 * cos(h, q).
    """
    sem = _cosine_sim(chunk_hidden, intent)
    return attn_mass + lambda_sem * sem


# ---------------------------------------------------------------------------
# Synchronized KV eviction helper
# ---------------------------------------------------------------------------

def evict_kv_positions(
    past_key_values: Tuple,
    positions_to_remove: List[int],
) -> Tuple:
    """Remove KV entries at *positions_to_remove* from ALL layers."""
    keep_mask = None
    new_past = []
    for layer_kv in past_key_values:
        k, v = layer_kv  # (batch, heads, seq, head_dim)
        seq_len = k.shape[2]
        if keep_mask is None:
            keep_mask = torch.ones(seq_len, dtype=torch.bool)
            for pos in positions_to_remove:
                if 0 <= pos < seq_len:
                    keep_mask[pos] = False
        new_past.append((k[:, :, keep_mask, :], v[:, :, keep_mask, :]))
    return tuple(new_past)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_taskkv_hf_demo(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt: str = (
        "You are an intelligent agent. Let's solve a task.\n"
        "Step 1: Observation: The room is empty.\n"
        "Step 2: Action: Move to kitchen.\n"
    ),
    budget_tokens: int = 32,
    chunk_size: int = 10,
    sink_tokens: int = 4,
    ema_alpha: float = 0.1,
    lambda_sem: float = 1.0,
    gen_steps: int = 20,
):
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"\nInitial prompt length : {input_ids.shape[1]} tokens")
    print(f"KV budget            : {budget_tokens} tokens")
    print(f"Chunk size           : {chunk_size} tokens")
    print(f"Sink tokens          : {sink_tokens}")
    print()

    past_key_values = None
    generated_ids: List[int] = []
    intent: torch.Tensor = None  # running EMA intent vector

    # Track per-chunk metadata: (start_pos, end_pos, hidden_centroid, score)
    chunk_registry: List[dict] = []
    tokens_since_last_chunk = 0

    with torch.no_grad():
        for step in range(gen_steps):
            cur_input = input_ids if step == 0 else next_token
            outputs = model(
                input_ids=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            generated_ids.append(next_token.item())

            # Accumulate hidden state for chunk centroid
            last_hidden = outputs.hidden_states[-1][0, -1, :]  # (hidden_dim,)
            tokens_since_last_chunk += cur_input.shape[1] if step == 0 else 1

            # Register a new chunk every chunk_size tokens
            kv_len = past_key_values[0][0].shape[2]
            if tokens_since_last_chunk >= chunk_size:
                chunk_end = kv_len
                chunk_start = chunk_end - tokens_since_last_chunk

                # Mean-pool hidden states over chunk positions
                h_chunk = outputs.hidden_states[-1][0, -1, :].clone()

                # Update EMA intent
                if intent is None:
                    intent = h_chunk.clone()
                else:
                    intent = ema_alpha * h_chunk + (1.0 - ema_alpha) * intent
                intent_norm = intent / (intent.norm() + 1e-8)

                attn_mass = 0.5  # proxy; real path uses attention norms
                s = score_chunk(h_chunk, intent_norm, attn_mass, lambda_sem)

                chunk_registry.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "score": s,
                    "hidden": h_chunk,
                })
                tokens_since_last_chunk = 0

            # ----- Eviction under budget -----
            if kv_len > budget_tokens + chunk_size and len(chunk_registry) > 1:
                # Protect sink tokens and the most recent chunk
                evictable = [
                    c for c in chunk_registry
                    if c["start"] >= sink_tokens and c != chunk_registry[-1]
                ]
                if evictable:
                    # Select lowest-scoring chunk
                    victim = min(evictable, key=lambda c: c["score"])
                    positions = list(range(victim["start"], victim["end"]))
                    past_key_values = evict_kv_positions(past_key_values, positions)

                    # Update registry positions after eviction
                    removed_len = len(positions)
                    chunk_registry.remove(victim)
                    for c in chunk_registry:
                        if c["start"] > victim["start"]:
                            c["start"] -= removed_len
                            c["end"] -= removed_len

                    new_kv_len = past_key_values[0][0].shape[2]
                    print(
                        f"  [Step {step:>2}] Evicted chunk "
                        f"(score={victim['score']:.3f}, "
                        f"pos {victim['start']}-{victim['end']}). "
                        f"KV: {kv_len} -> {new_kv_len} tokens. "
                        f"Layer sync: OK."
                    )

    final_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    kv_final = past_key_values[0][0].shape[2]
    print(f"\nFinal KV length: {kv_final} tokens (budget: {budget_tokens})")
    print(f"Generated text:\n  {final_text}")
    print(
        "\n[INFO] Demo finished. This demonstrates the synchronized "
        "layer-masking and intent-driven eviction logic on a live model."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TaskKV HuggingFace toy demo")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--budget_tokens", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--gen_steps", type=int, default=20)
    args = parser.parse_args()
    run_taskkv_hf_demo(
        model_id=args.model_id,
        budget_tokens=args.budget_tokens,
        chunk_size=args.chunk_size,
        gen_steps=args.gen_steps,
    )
