"""
Data types for TaskKV trajectory representation.

A trajectory  T = {s_1, ..., s_t}  is a sequence of semantic steps,
where each step corresponds to one Thought / Action / Observation
block in the ReAct-style agent interaction.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(order=True)
class Step:
    """
    A single semantic step in the agent trajectory.

    Fields used by the min-heap in MemoryManager (Algorithm 1):
      - score  :  f(s_i, q_t)  computed by the scorer
      - timestamp  :  creation order (for tie-breaking and re-sorting)

    Auxiliary fields (not compared):
      - name       :  human-readable label
      - step_type  :  one of {"system", "goal", "thought", "action",
                       "observation", "error"}
      - embedding  :  h_i  centroid vector (key-state pooling, Sec. 3.2.2)
      - token_span :  (start, end) positions in the original sequence
      - token_length : number of retained KV tokens consumed by this step
      - pivot      : scalar parser feature Psi(s_i)
    """
    score: float
    timestamp: int = field(compare=False)
    name: str = field(compare=False, default="")
    step_type: str = field(compare=False, default="observation")
    embedding: Optional[List[float]] = field(compare=False, default=None, repr=False)
    token_span: Optional[tuple] = field(compare=False, default=None, repr=False)
    token_length: int = field(compare=False, default=1)
    pivot: float = field(compare=False, default=0.0)

    def __repr__(self):
        return (
            f"Step(t={self.timestamp}, type={self.step_type}, "
            f"tokens={self.token_length}, score={self.score:.4f})"
        )


@dataclass
class TrajectorySegment:
    """
    Raw token segment before it is converted into a scored Step.
    Used by the segmentation module (Sec. 3.2.2) to accumulate tokens
    until a semantic boundary or the fail-safe stride Delta_max is hit.
    """
    tokens: List[int] = field(default_factory=list)
    start_pos: int = 0
    step_type: str = "observation"

    @property
    def length(self) -> int:
        return len(self.tokens)

    def reset(self, new_start: int):
        self.tokens = []
        self.start_pos = new_start
        self.step_type = "observation"
