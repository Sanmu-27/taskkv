"""
Budgeted KV-cache memory manager (Algorithm 1 in the paper).

Maintains a min-heap of retained semantic steps under an exact token budget.
Each step carries a variable token_length. New steps are inserted, then the
lowest-scoring steps are evicted until the retained token sum falls below the
configured budget.

A periodic lazy re-ranking pass updates scores with the current intent vector
q_t to handle intent drift.
"""

import heapq
from typing import List, Tuple

from .types import Step


class MemoryManager:
    """Heap-based semantic-step retention with strict token accounting."""

    def __init__(self, budget_tokens: int):
        self.budget_tokens = int(budget_tokens)
        self._heap: List[Step] = []
        self._eviction_log: List[Step] = []
        self._retained_tokens = 0

    def add(self, step: Step) -> Tuple[str, List[Step]]:
        """
        Try to insert *step* into memory.

        Returns:
            (status_message, evicted_steps)
        """
        heapq.heappush(self._heap, step)
        self._retained_tokens += self._length(step)

        evicted: List[Step] = []
        while self._retained_tokens > self.budget_tokens and self._heap:
            victim = heapq.heappop(self._heap)
            self._retained_tokens -= self._length(victim)
            self._eviction_log.append(victim)
            evicted.append(victim)

        if not evicted:
            return "added", []
        if len(evicted) == 1 and evicted[0].timestamp == step.timestamp:
            return "pruned", evicted
        return "rebalanced", evicted

    def rerank(self, score_fn):
        """
        Re-score every step in the heap with the current intent, then re-heapify.
        *score_fn* maps Step -> float.
        """
        for s in self._heap:
            s.score = score_fn(s)
        heapq.heapify(self._heap)

    def snapshot(self) -> List[Step]:
        """Current memory contents, sorted chronologically."""
        return sorted(self._heap, key=lambda s: s.timestamp)

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def retained_tokens(self) -> int:
        return self._retained_tokens

    @property
    def is_full(self) -> bool:
        return self._retained_tokens >= self.budget_tokens

    @staticmethod
    def _length(step: Step) -> int:
        return max(1, int(step.token_length))
