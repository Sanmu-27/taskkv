"""
TaskKV configuration.

Default values are aligned with the paper's reported settings where possible.
The lightweight simulation config remains intentionally small and CPU friendly.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskKVConfig:
    hidden_dim: int = 8192
    num_layers: int = 80
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128

    lambda_semantic: float = 1.0
    lambda_noise: float = 1.2
    ema_alpha: float = 0.1

    budget_ratio: float = 0.2
    max_stride: int = 64
    recheck_interval: int = 16

    entropy_threshold: float = 0.8
    sink_tokens: int = 4

    scorer_hidden: int = 256
    scorer_layers: int = 3
    scorer_dropout: float = 0.1

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    batch_size: int = 32
    ranknet_margin: float = 1.0

    device: str = "cuda"
    seed: int = 42

    model_path: Optional[str] = None
    scorer_checkpoint: Optional[str] = None


@dataclass
class SimulationConfig:
    """Lightweight config for the CPU-only demo and unit tests."""

    hidden_dim: int = 128
    num_layers: int = 32
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 16

    lambda_semantic: float = 1.0
    lambda_noise: float = 1.2
    ema_alpha: float = 0.1

    budget_ratio: float = 0.2
    max_stride: int = 64
    recheck_interval: int = 16

    entropy_threshold: float = 0.8
    sink_tokens: int = 4

    scorer_hidden: int = 64
    scorer_layers: int = 3
    scorer_dropout: float = 0.0

    device: str = "cpu"
    seed: int = 42
