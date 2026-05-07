"""
Train the TaskKV step-importance scorer with RankNet on signed labels.

The deployed residual scorer takes

    [h_i ; q_t ; h_i * q_t ; Psi(s_i)] -> scalar utility score

and is trained with a pairwise RankNet objective on signed first-order
action-utility labels produced by collect_trajectories.py. LOO and SMI
diagnostics are not training targets in this artifact.

Usage:
    python scripts/train_scorer.py \
        --config configs/<scorer_config>.yaml \
        --trajectory_dir <path-to-trajectories> \
        --output_dir checkpoints/scorer
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


def load_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("[ERROR] PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_trajectories(trajectory_dir: str) -> list:
    traj_dir = Path(trajectory_dir)
    if not traj_dir.exists():
        print(f"[ERROR] Trajectory directory not found: {trajectory_dir}")
        sys.exit(1)

    files = sorted(traj_dir.glob("trajectory_*.json"))
    if not files:
        print(f"[ERROR] No trajectory files found in {trajectory_dir}")
        sys.exit(1)

    trajectories = []
    for file_path in files:
        with open(file_path) as fp:
            trajectories.append(json.load(fp))

    print(f"[INFO] Loaded {len(trajectories)} trajectories from {trajectory_dir}")
    return trajectories


class ScorerMLP:
    """The residual scorer architecture from Eq. 7."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def build_torch_module(self):
        try:
            import torch.nn as nn
        except ImportError:
            print("[ERROR] PyTorch not installed. Run: pip install torch")
            sys.exit(1)

        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
        )


def ranknet_loss(score_pos, score_neg):
    try:
        import torch.nn.functional as F
    except ImportError:
        print("[ERROR] PyTorch required for training.")
        sys.exit(1)
    return F.softplus(-(score_pos - score_neg)).mean()


def _step_embedding(step, hidden_dim, torch):
    if "embedding" in step:
        return torch.tensor(step["embedding"], dtype=torch.float32)
    torch.manual_seed(hash(step.get("text", "")) % (2**31))
    return torch.randn(hidden_dim)


def _pivot_value(step):
    return float(step.get("pivot", 0.0))


def _intent_vector(step_vecs, alpha, torch):
    q = torch.zeros_like(step_vecs[0])
    for vec in step_vecs:
        q = alpha * vec + (1.0 - alpha) * q
    return q / (q.norm() + 1e-8)


def _build_pairs(trajectories, hidden_dim, alpha, pairs_per_trajectory, torch):
    pairs = []
    for traj in trajectories:
        steps = traj.get("steps", [])
        labels = traj.get("signed_utility")
        if labels is None or len(labels) != len(steps) or len(steps) < 2:
            continue

        step_vecs = [_step_embedding(step, hidden_dim, torch) for step in steps]
        q = _intent_vector(step_vecs, alpha, torch)

        indexed = list(enumerate(labels))
        indexed.sort(key=lambda x: x[1], reverse=True)
        usable = [item for item in indexed if abs(item[1]) > 1e-8]
        if len(usable) < 2:
            continue

        pos_candidates = usable[: max(1, len(usable) // 2)]
        neg_candidates = list(reversed(usable[max(1, len(usable) // 2) :]))
        if not neg_candidates:
            neg_candidates = list(reversed(usable[1:]))

        max_pairs = min(pairs_per_trajectory, len(pos_candidates) * len(neg_candidates))
        count = 0
        for pos_idx, pos_label in pos_candidates:
            for neg_idx, neg_label in neg_candidates:
                if pos_label <= neg_label:
                    continue
                pairs.append(
                    (
                        step_vecs[pos_idx],
                        q.clone(),
                        _pivot_value(steps[pos_idx]),
                        step_vecs[neg_idx],
                        q.clone(),
                        _pivot_value(steps[neg_idx]),
                    )
                )
                count += 1
                if count >= max_pairs:
                    break
            if count >= max_pairs:
                break
    return pairs


def train(config: dict, trajectories: list, output_dir: str):
    try:
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
    except ImportError:
        print("[ERROR] PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    model_cfg = config.get("model", {})
    hidden_dim = model_cfg.get("hidden_dim", config.get("hidden_dim", 8192))
    scorer_cfg = config.get("scorer", {})
    model = ScorerMLP(
        input_dim=3 * hidden_dim + 1,
        hidden_dim=scorer_cfg.get("hidden_dim", 256),
        dropout=scorer_cfg.get("dropout", 0.1),
    ).build_torch_module()

    device = torch.device(config.get("device", "cuda"))
    model = model.to(device)

    training_cfg = config.get("training", {})
    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", config.get("learning_rate", 1e-4)),
        weight_decay=training_cfg.get("weight_decay", config.get("weight_decay", 0.01)),
    )
    num_epochs = training_cfg.get("num_epochs", config.get("num_epochs", 5))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    alpha = config.get("ema_alpha", 0.1)
    pairs = _build_pairs(
        trajectories,
        hidden_dim=hidden_dim,
        alpha=alpha,
        pairs_per_trajectory=config.get("pairs_per_trajectory", 50),
        torch=torch,
    )
    if not pairs:
        print("[ERROR] No valid training pairs. Check signed_utility labels.")
        sys.exit(1)

    print(f"[INFO] Built {len(pairs)} training pairs")
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = training_cfg.get("batch_size", config.get("batch_size", 32))
    best_loss = float("inf")

    for epoch in range(num_epochs):
        random.shuffle(pairs)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]

            h_pos = torch.stack([item[0] for item in batch]).to(device)
            q_pos = torch.stack([item[1] for item in batch]).to(device)
            p_pos = torch.tensor([[item[2]] for item in batch], dtype=torch.float32, device=device)

            h_neg = torch.stack([item[3] for item in batch]).to(device)
            q_neg = torch.stack([item[4] for item in batch]).to(device)
            p_neg = torch.tensor([[item[5]] for item in batch], dtype=torch.float32, device=device)

            inp_pos = torch.cat([h_pos, q_pos, h_pos * q_pos, p_pos], dim=-1)
            inp_neg = torch.cat([h_neg, q_neg, h_neg * q_neg, p_neg], dim=-1)

            score_pos = model(inp_pos).squeeze(-1)
            score_neg = model(inp_neg).squeeze(-1)
            loss = ranknet_loss(score_pos, score_neg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, "scorer_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best checkpoint -> {ckpt_path}")

    final_path = os.path.join(output_dir, "scorer_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Training complete. Final checkpoint -> {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TaskKV scorer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--trajectory_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    trajectories = load_trajectories(args.trajectory_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    train(config, trajectories, args.output_dir)


if __name__ == "__main__":
    main()
