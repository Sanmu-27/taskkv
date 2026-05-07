"""
Portable tensor operations for CPU-only simulation / demo.

When running with a real model, these are replaced by torch.Tensor
operations via the standard forward pass. This module exists only
to make the scoring logic inspectable without a GPU.
"""

import math
import random
from typing import List, Tuple


class Tensor:
    """Minimal vector wrapper — pure-Python, no external deps."""

    __slots__ = ("data", "shape")

    def __init__(self, data: List[float], shape: Tuple[int, ...]):
        self.data = data
        self.shape = shape
        expected = 1
        for d in shape:
            expected *= d
        assert len(data) == expected, f"shape {shape} vs len {len(data)}"

    # ── constructors ────────────────────────────────────────────────────
    @classmethod
    def zeros(cls, shape: Tuple[int, ...]):
        n = 1
        for d in shape:
            n *= d
        return cls([0.0] * n, shape)

    @classmethod
    def randn(cls, shape: Tuple[int, ...], seed=None):
        if seed is not None:
            random.seed(seed)
        n = 1
        for d in shape:
            n *= d
        return cls([random.gauss(0, 1) for _ in range(n)], shape)

    # ── linear algebra ──────────────────────────────────────────────────
    def dot(self, other: "Tensor") -> float:
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.data))

    def normalize(self) -> "Tensor":
        n = self.norm()
        if n < 1e-9:
            return self
        return Tensor([x / n for x in self.data], self.shape)

    def hadamard(self, other: "Tensor") -> "Tensor":
        """Element-wise (Hadamard) product  h ⊙ q."""
        return Tensor([a * b for a, b in zip(self.data, other.data)], self.shape)

    def concat(self, *others: "Tensor") -> "Tensor":
        """Concatenate along the flat dimension — used to build [h; q; h⊙q]."""
        out = list(self.data)
        total = self.shape[0]
        for o in others:
            out.extend(o.data)
            total += o.shape[0]
        return Tensor(out, (total,))

    # ── arithmetic ──────────────────────────────────────────────────────
    def __add__(self, other: "Tensor"):
        return Tensor([a + b for a, b in zip(self.data, other.data)], self.shape)

    def __mul__(self, scalar: float):
        return Tensor([x * scalar for x in self.data], self.shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"


def sigmoid(x: float) -> float:
    """Numerically-stable scalar sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


