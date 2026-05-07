"""
TaskKV scoring module -- Eq. 7 in the paper.

    f(s_i, q_t) = phi_attn(s_i)
                + lambda_1 * h_i^T q_t
                - lambda_2 * sigma(MLP_phi([h_i; q_t; h_i * q_t; Psi(s_i)]))

In benchmark reproduction the MLP weights are loaded from a trained checkpoint.
This reference implementation also provides a deterministic fallback projection
so that the scoring logic, entropy switch, and heap-based retention can be
inspected without requiring model weights or a GPU.
"""

from ..utils.tensor import Tensor, sigmoid


class _FallbackMLP:
    """
    CPU-only fallback for the denoising MLP D_phi.

    The benchmark path loads a torch checkpoint via load_weights(). The fallback
    path uses a fixed projection only for local sanity checks.
    """

    def __init__(self, hidden_dim: int, mlp_hidden: int = 256, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.input_dim = 3 * hidden_dim + 1
        self.mlp_hidden = mlp_hidden
        self._proj = Tensor.randn((self.input_dim,), seed=seed).normalize()

    def load_weights(self, path: str):
        """Load trained MLP weights from a PyTorch checkpoint."""
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Scorer checkpoint not found at '{path}'. "
                "Train one with: python scripts/train_scorer.py"
            )

        import torch
        import torch.nn as nn

        layers = []
        in_dim = self.input_dim
        for hidden in (self.mlp_hidden, 64):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))

        self._torch_module = nn.Sequential(*layers)
        self._torch_module.load_state_dict(torch.load(path, map_location="cpu"))
        self._torch_module.eval()
        self._use_torch = True

    def forward(self, h, q, pivot: float = 0.0):
        """Return the denoising logit for [h_i; q_t; h_i*q_t; Psi(s_i)]."""
        pivot_t = Tensor([float(pivot)], (1,))
        inp = h.concat(q, h.hadamard(q), pivot_t)

        if getattr(self, "_use_torch", False):
            import torch

            inp_t = torch.tensor(inp.data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return self._torch_module(inp_t).item()

        return inp.dot(self._proj)


class TaskKVScorer:
    """
    Implements the modular proxy scoring function from Eq. 7.

    Parameters
    ----------
    hidden_dim : int
        Dimension of step centroid vectors h_i.
    lambda_semantic : float
        Weight lambda_1 for semantic alignment.
    lambda_noise : float
        Weight lambda_2 for the learned denoising penalty.
    """

    def __init__(
        self,
        hidden_dim: int,
        lambda_semantic: float = 1.0,
        lambda_noise: float = 1.2,
        scorer_hidden: int = 256,
        seed: int = 42,
    ):
        self.lambda_1 = lambda_semantic
        self.lambda_2 = lambda_noise
        self._mlp = _FallbackMLP(hidden_dim, scorer_hidden, seed=seed)

    def forward(self, h_i: Tensor, q_t: Tensor, attn_norm: float, pivot: float = 0.0) -> dict:
        """
        Score a single semantic step.

        Args:
            h_i: normalized step centroid.
            q_t: normalized EMA intent vector.
            attn_norm: length-normalized per-step attention mass.
            pivot: scalar parser feature Psi(s_i).

        Returns:
            A dict with keys "score", "sem", "noise", and "pivot".
        """
        sem = h_i.dot(q_t)
        noise_logit = self._mlp.forward(h_i, q_t, pivot=pivot)
        noise_prob = sigmoid(noise_logit)
        score = attn_norm + self.lambda_1 * sem - self.lambda_2 * noise_prob
        return {"score": score, "sem": sem, "noise": noise_prob, "pivot": pivot}

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs):
        """Load a trained scorer checkpoint."""
        scorer = cls(**kwargs)
        scorer._mlp.load_weights(path)
        return scorer
