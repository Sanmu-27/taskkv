"""
Baseline KV-cache compression policies used for comparison in Table 1.

Each baseline implements the same interface:
    select(scores, budget) -> List[int]   (indices to retain)

These are thin wrappers around the upstream implementations.  For full
reproduction, install the respective packages and point to the model
checkpoint.  See README §Baseline Comparison.
"""

from typing import List


class H2OPolicy:
    """
    Heavy-Hitter Oracle (H2O) — Zhang et al. 2023.
    Retains tokens with highest accumulated attention mass.
    Prompt-protected variant force-retains the first 10% of tokens (referred to as 'H2O' in the paper).
    """

    def __init__(self, protect_ratio: float = 0.1):
        self.protect_ratio = protect_ratio

    def select(self, attn_scores: List[float], budget: int) -> List[int]:
        n = len(attn_scores)
        protect_end = int(n * self.protect_ratio)
        protected = list(range(protect_end))
        remaining_budget = budget - len(protected)

        if remaining_budget <= 0:
            return protected[:budget]

        candidates = list(range(protect_end, n))
        candidates.sort(key=lambda i: attn_scores[i], reverse=True)
        selected = protected + candidates[:remaining_budget]
        return sorted(selected)


class StreamingLLMPolicy:
    """
    StreamingLLM — Xiao et al. 2023.
    Keeps initial sink tokens + a sliding window of the most recent tokens.
    """

    def __init__(self, sink_tokens: int = 4):
        self.sink_tokens = sink_tokens

    def select(self, seq_len: int, budget: int) -> List[int]:
        sink = list(range(min(self.sink_tokens, seq_len)))
        window_budget = budget - len(sink)
        if window_budget <= 0:
            return sink[:budget]
        window_start = max(len(sink), seq_len - window_budget)
        window = list(range(window_start, seq_len))
        return sorted(set(sink + window))


class SnapKVPolicy:
    """
    SnapKV — Li et al. 2024.
    Clusters attention patterns to identify important observation windows.

    When per-layer attention maps are available (from a live model forward
    pass), the policy pools attention mass within sliding observation windows
    and selects the top-budget windows.  The ``select`` method below
    implements this pooled-window selection given pre-computed per-position
    attention scores (one score per KV position, aggregated across heads
    and layers by the caller).

    For the full upstream SnapKV integration, install the snapkv package
    and pass per-layer attention maps directly.
    """

    def __init__(self, window_size: int = 64, sink_tokens: int = 4):
        self.window_size = window_size
        self.sink_tokens = sink_tokens

    def select(self, attn_scores: List[float], budget: int) -> List[int]:
        """Select token indices to retain under the given budget.

        Args:
            attn_scores: per-position pooled attention scores (length = seq_len).
            budget: number of tokens to retain.
        """
        n = len(attn_scores)
        if budget >= n:
            return list(range(n))

        # Always retain sink tokens
        sink_end = min(self.sink_tokens, n)
        protected = list(range(sink_end))
        remaining_budget = budget - len(protected)
        if remaining_budget <= 0:
            return protected[:budget]

        # Pool scores into observation windows
        candidates = list(range(sink_end, n))
        windows = []
        for start in range(0, len(candidates), self.window_size):
            window_indices = candidates[start : start + self.window_size]
            window_score = sum(attn_scores[i] for i in window_indices) / len(window_indices)
            windows.append((window_score, window_indices))

        # Rank windows by pooled score, greedily add until budget is met
        windows.sort(key=lambda w: w[0], reverse=True)
        selected = list(protected)
        for _, indices in windows:
            if len(selected) + len(indices) <= budget:
                selected.extend(indices)
            else:
                needed = budget - len(selected)
                ranked = sorted(indices, key=lambda i: attn_scores[i], reverse=True)
                selected.extend(ranked[:needed])
                break
        return sorted(selected)


class StrongHeuristicPolicy:
    """
    Strong training-free heuristic baseline (Table 1).

    Allocates the budget as: 10% system-prompt protection, 30% recent
    sliding window, 60% semantic top-k retrieval (cosine similarity
    between step centroids and the current intent vector).
    """

    def __init__(
        self,
        protect_ratio: float = 0.1,
        recent_ratio: float = 0.3,
        sink_tokens: int = 4,
    ):
        self.protect_ratio = protect_ratio
        self.recent_ratio = recent_ratio
        self.sink_tokens = sink_tokens

    def select(
        self,
        semantic_scores: List[float],
        budget: int,
    ) -> List[int]:
        """Select token indices to retain under the given budget.

        Args:
            semantic_scores: per-position cosine-similarity scores to intent.
            budget: number of tokens to retain.
        """
        n = len(semantic_scores)
        if budget >= n:
            return list(range(n))

        # Prompt protection
        protect_end = max(self.sink_tokens, int(n * self.protect_ratio))
        protected = list(range(min(protect_end, n)))

        # Recent window
        recent_budget = int(budget * self.recent_ratio)
        recent_start = max(protect_end, n - recent_budget)
        recent = list(range(recent_start, n))

        # Semantic top-k for the remaining budget
        already = set(protected) | set(recent)
        remaining_budget = budget - len(already)
        candidates = [(i, semantic_scores[i]) for i in range(n) if i not in already]
        candidates.sort(key=lambda x: x[1], reverse=True)
        semantic = [i for i, _ in candidates[:max(0, remaining_budget)]]

        return sorted(set(protected) | set(recent) | set(semantic))


class AdaKVPolicy:
    """
    AdaKV — Adaptive KV-cache compression (Feng et al. 2024).

    Pools attention budgets across layers adaptively: layers with
    higher attention entropy receive a larger share of the budget.
    At the token level within each layer, heavy-hitter selection
    is used (same as H2O).

    This wrapper implements the token-level selection given
    pre-computed per-position attention scores and a layer-specific
    budget determined by the adaptive pooling.
    """

    def __init__(self, protect_ratio: float = 0.1, recent_ratio: float = 0.2):
        self.protect_ratio = protect_ratio
        self.recent_ratio = recent_ratio

    def select(
        self,
        attn_scores: List[float],
        budget: int,
    ) -> List[int]:
        """Select token indices to retain under the given budget.

        The adaptive per-layer budget is determined externally by the
        serving engine; this method implements the within-layer selection.
        """
        n = len(attn_scores)
        if budget >= n:
            return list(range(n))

        protect_end = int(n * self.protect_ratio)
        protected = list(range(min(protect_end, n)))

        recent_count = int(budget * self.recent_ratio)
        recent_start = max(protect_end, n - recent_count)
        recent = list(range(recent_start, n))

        already = set(protected) | set(recent)
        remaining = budget - len(already)
        candidates = [
            (i, attn_scores[i]) for i in range(protect_end, recent_start)
            if i not in already
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [i for i, _ in candidates[:max(0, remaining)]]

        return sorted(set(protected) | set(recent) | set(selected))
