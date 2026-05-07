"""
Synchronized KV cache pruning across transformer layers.

Design rationale (Sec. 3.2.4):
  All L layers evict the *same* token positions to preserve
  residual-stream coherence.  Layer-specific pruning degrades
  SR by -4.3% because removing a token at layer l-1 breaks
  the identity path expected by layer l.

In PagedAttention-style serving engines (Appendix C.3), physical
eviction can be replaced by block-table remapping to avoid memcpy
overhead.
"""

from typing import List


class SynchronizedKVCache:
    """
    Multi-layer KV cache with synchronized eviction (Sec. 3.2.4).

    This class manages per-layer KV caches and enforces that eviction
    decisions are broadcast identically across all L layers, preserving
    residual-stream coherence.

    In PagedAttention-style serving engines (Appendix C.3), physical
    eviction is replaced by block-table remapping to avoid memcpy
    overhead; the logical semantics are the same.
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._layers = [[] for _ in range(num_layers)]

    def append_token(self, layer_id: int, key, value):
        """Append a single token's KV pair to the specified layer."""
        self._layers[layer_id].append((key, value))

    def evict(self, step_indices: List[int]):
        """
        Evict KV entries at *step_indices* from ALL layers simultaneously
        (synchronized masking).
        """
        remove = set(step_indices)
        for lid in range(self.num_layers):
            self._layers[lid] = [
                kv for idx, kv in enumerate(self._layers[lid])
                if idx not in remove
            ]

    def retain_only(self, keep_indices: List[int]):
        """Keep only the specified indices, evict everything else."""
        keep = set(keep_indices)
        for lid in range(self.num_layers):
            self._layers[lid] = [
                kv for idx, kv in enumerate(self._layers[lid])
                if idx in keep
            ]

    def get_layer_cache(self, layer_id: int):
        """Return the (key, value) list for a specific layer."""
        return self._layers[layer_id]

    @property
    def seq_len(self) -> int:
        return len(self._layers[0]) if self._layers else 0

    def memory_bytes(self) -> int:
        """Estimate VRAM usage assuming float16 storage."""
        # 2 (K+V) * num_layers * seq_len * num_kv_heads * head_dim * 2 bytes
        return 2 * self.num_layers * self.seq_len * self.num_kv_heads * self.head_dim * 2
