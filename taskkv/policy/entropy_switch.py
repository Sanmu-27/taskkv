"""
Adaptive Density Switching  —  Eq. 8 in the paper.

Monitors the entropy of the utility-score distribution over the
memory buffer.  When the distribution is "flat" (H(f) >= tau), the
scorer cannot reliably distinguish important steps, so the system
falls back to StreamingLLM-style sliding-window retention.

    Mode = { TaskKV Pruning    if H(f) < tau   (sparse regime)
           { StreamingLLM      if H(f) >= tau  (dense regime)
"""

import math
from typing import List


class EntropySafeguard:
    """
    Density-aware switch between TaskKV scoring and sliding-window.

    Parameters
    ----------
    tau : float
        Entropy threshold (default 0.8).  Above this, scoring is
        considered unreliable and the system reverts to recency.
    sink_tokens : int
        Number of initial "attention sink" tokens always retained
        when in StreamingLLM mode.
    """

    def __init__(self, tau: float = 0.8, sink_tokens: int = 4):
        self.tau = tau
        self.sink_tokens = sink_tokens
        self._last_entropy: float = 0.0
        self._mode: str = "taskkv"

    def compute_entropy(self, scores: List[float]) -> float:
        """
        Compute normalized entropy of the softmax score distribution.

        H(f) = -sum( p_i * log(p_i) )  /  log(N)

        where p_i = exp(f_i) / sum(exp(f_j)).
        """
        if len(scores) <= 1:
            return 0.0

        # numerically stable softmax
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps]

        entropy = 0.0
        for p in probs:
            if p > 1e-12:
                entropy -= p * math.log(p)

        # normalize to [0, 1]
        max_entropy = math.log(len(scores))
        if max_entropy > 0:
            entropy /= max_entropy

        self._last_entropy = entropy
        return entropy

    def should_use_taskkv(self, scores: List[float]) -> bool:
        """Return True if TaskKV scoring should be active (sparse regime)."""
        h = self.compute_entropy(scores)
        self._mode = "taskkv" if h < self.tau else "streamingllm"
        return h < self.tau

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def last_entropy(self) -> float:
        return self._last_entropy
