"""
TaskKV online streaming simulator (Algorithm 1 in the paper).

This module ties together the scorer, memory manager, entropy safeguard, and
synchronized KV cache to simulate the online inference loop on synthetic
trajectories.

For benchmark reproduction, the equivalent logic lives inside the serving
integration described in the paper. This simulator exists so reviewers can
inspect the algorithmic flow without GPUs or external model weights.
"""

from typing import Dict, List, Optional

from .core.memory_manager import MemoryManager
from .core.types import Step
from .policy.entropy_switch import EntropySafeguard
from .policy.scorer import TaskKVScorer
from .utils.config import SimulationConfig
from .utils.tensor import Tensor


_TYPE_SEEDS = {
    "system": 10,
    "goal": 20,
    "thought": 30,
    "action": 40,
    "observation": 50,
    "error": 999,
}


def _make_step_vector(dim: int, step_type: str, step_id: int) -> Tensor:
    """
    Generate a deterministic pseudo-embedding for a step.

    In production this is replaced by key-state pooling. For the demo, error
    steps get a detectable signature so the fallback denoising head can
    separate them from useful steps.
    """
    base_seed = _TYPE_SEEDS.get(step_type, 0)
    h = Tensor.randn((dim,), seed=base_seed + step_id)
    if step_type == "error":
        for i in range(min(8, dim)):
            h.data[i] = abs(h.data[i]) + 1.0
    else:
        for i in range(min(8, dim)):
            h.data[i] = -(abs(h.data[i]) + 0.5)
    return h.normalize()


def _pivot_feature(step_type: str) -> float:
    """Parser feature Psi(s_i): fires on explicit strategy-pivot steps."""
    return 1.0 if step_type in {"pivot", "goal"} else 0.0


class TaskKVSimulator:
    """Simulates TaskKV on a synthetic trajectory."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        self.scorer = TaskKVScorer(
            hidden_dim=self.cfg.hidden_dim,
            lambda_semantic=self.cfg.lambda_semantic,
            lambda_noise=self.cfg.lambda_noise,
            scorer_hidden=self.cfg.scorer_hidden,
            seed=self.cfg.seed,
        )
        self.safeguard = EntropySafeguard(
            tau=self.cfg.entropy_threshold,
            sink_tokens=self.cfg.sink_tokens,
        )
        self._budget_ratio = self.cfg.budget_ratio
        self._budget = None
        self.memory = None
        self._intent = Tensor.zeros((self.cfg.hidden_dim,))
        self._step_counter = 0
        self._history: List[Dict] = []

    def process_step(
        self,
        name: str,
        step_type: str = "observation",
        attn_bias: float = 0.5,
        token_length: int = 1,
    ) -> Dict:
        """Process one semantic step through the TaskKV pipeline."""
        self._step_counter += 1

        if self.memory is None:
            self._budget = max(2, int(20 * self._budget_ratio))
            self.memory = MemoryManager(budget_tokens=self._budget)

        h_i = _make_step_vector(self.cfg.hidden_dim, step_type, self._step_counter)
        pivot = _pivot_feature(step_type)

        alpha = self.cfg.ema_alpha
        self._intent = (h_i * alpha + self._intent * (1.0 - alpha)).normalize()

        metrics = self.scorer.forward(h_i, self._intent, attn_bias, pivot=pivot)
        current_scores = [s["score"] for s in self._history] + [metrics["score"]]
        self.safeguard.should_use_taskkv(current_scores)

        step = Step(
            score=metrics["score"],
            timestamp=self._step_counter,
            name=name,
            step_type=step_type,
            embedding=h_i.data,
            token_length=token_length,
            pivot=pivot,
        )
        status, evicted = self.memory.add(step)

        if (
            self._step_counter % self.cfg.recheck_interval == 0
            and self.memory is not None
            and self.memory.size > 0
        ):
            intent_snapshot = self._intent

            def _rescore(s: Step) -> float:
                s_vec = Tensor(list(s.embedding), (self.cfg.hidden_dim,))
                return self.scorer.forward(
                    s_vec, intent_snapshot, attn_bias, pivot=s.pivot
                )["score"]

            self.memory.rerank(_rescore)

        record = {
            "name": name,
            "type": step_type,
            "score": metrics["score"],
            "sem": metrics["sem"],
            "noise": metrics["noise"],
            "pivot": metrics["pivot"],
            "status": status,
            "evicted": [s.name for s in evicted],
            "mode": self.safeguard.mode,
            "entropy": self.safeguard.last_entropy,
            "retained_tokens": self.memory.retained_tokens,
        }
        self._history.append(record)
        return record

    def get_retained_memory(self) -> List[Dict]:
        """Return the current memory snapshot as a list of dicts."""
        if self.memory is None:
            return []
        snapshot = self.memory.snapshot()
        return [
            {
                "name": s.name,
                "type": s.step_type,
                "score": s.score,
                "t": s.timestamp,
                "tokens": s.token_length,
                "pivot": s.pivot,
            }
            for s in snapshot
        ]

    @property
    def history(self) -> List[Dict]:
        return list(self._history)
