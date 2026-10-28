"""Consensus aggregation over per-view class probabilities.

This module is deliberately framework-agnostic: it takes a matrix of
per-view probabilities (NumPy array or PyTorch tensor) and returns the
consensus summary. The semantics intentionally match the inference-time
sibling library ViewConsensus, so a model trained with ViewTrainer and a
model wrapped by ViewConsensus report uncertainty the same way:

* consensus probability = mean of the per-view probabilities;
* uncertainty = population standard deviation of the winning class's
  probability across views;
* agreement = fraction of views whose own argmax equals the consensus
  label;
* every view's vote is retained, never averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

_ROW_SUM_TOL = 1e-3


@dataclass
class ViewVote:
    """One view's own prediction."""

    label_index: int
    probability: float  # that view's probability for its own argmax label


@dataclass
class ConsensusResult:
    """Consensus summary for a single sample."""

    probabilities: np.ndarray  # [C] consensus (mean) probabilities
    label_index: int
    probability: float  # consensus probability of the winning class
    uncertainty: float  # population std of the winning class across views
    class_uncertainty: np.ndarray  # [C] per-class std across views
    agreement: float  # fraction of views voting for the winning class
    votes: List[ViewVote] = field(default_factory=list)


def _as_numpy(view_probs: ArrayLike) -> np.ndarray:
    if isinstance(view_probs, torch.Tensor):
        arr = view_probs.detach().cpu().numpy()
    else:
        arr = np.asarray(view_probs)
    return np.asarray(arr, dtype=np.float64)


def validate_view_probabilities(view_probs: ArrayLike) -> np.ndarray:
    """Validate a ``[V, C]`` per-view probability matrix; return float64 NumPy."""
    arr = _as_numpy(view_probs)
    if arr.ndim != 2:
        raise ValueError(
            f"view probabilities must be a 2D [views, classes] matrix, "
            f"got shape {arr.shape}"
        )
    n_views, n_classes = arr.shape
    if n_views < 1:
        raise ValueError("at least one view is required")
    if n_classes < 2:
        raise ValueError(f"at least two classes are required, got {n_classes}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("view probabilities must be finite")
    if np.any(arr < 0):
        raise ValueError("view probabilities must be non-negative")
    row_sums = arr.sum(axis=1)
    if not np.all(np.abs(row_sums - 1.0) <= _ROW_SUM_TOL):
        raise ValueError(
            "each view's probabilities must sum to 1 "
            f"(tolerance {_ROW_SUM_TOL}); got row sums {row_sums}"
        )
    return arr


def consensus_from_probabilities(view_probs: ArrayLike) -> ConsensusResult:
    """Aggregate a ``[V, C]`` matrix of per-view probabilities.

    Returns a :class:`ConsensusResult` with the mean-probability
    consensus, the winning class's spread across views as uncertainty,
    the view agreement rate, and every view's retained vote.
    """
    arr = validate_view_probabilities(view_probs)
    consensus = arr.mean(axis=0)
    label_index = int(np.argmax(consensus))
    class_uncertainty = arr.std(axis=0)  # population std (ddof=0)
    votes = [
        ViewVote(label_index=int(np.argmax(row)), probability=float(np.max(row)))
        for row in arr
    ]
    agreement = float(np.mean([v.label_index == label_index for v in votes]))
    return ConsensusResult(
        probabilities=consensus,
        label_index=label_index,
        probability=float(consensus[label_index]),
        uncertainty=float(class_uncertainty[label_index]),
        class_uncertainty=class_uncertainty,
        agreement=agreement,
        votes=votes,
    )


def consensus_batch(view_probs: ArrayLike) -> List[ConsensusResult]:
    """Aggregate a ``[N, V, C]`` tensor of per-view probabilities, per sample."""
    arr = _as_numpy(view_probs)
    if arr.ndim != 3:
        raise ValueError(
            f"batched view probabilities must be 3D [samples, views, classes], "
            f"got shape {arr.shape}"
        )
    return [consensus_from_probabilities(arr[i]) for i in range(arr.shape[0])]
