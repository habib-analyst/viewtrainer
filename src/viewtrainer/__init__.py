"""ViewTrainer: consensus training for PyTorch classifiers.

Wrap any ``nn.Module`` classifier so its augmented views are trained to
agree, then predict with the same panel: consensus probability,
view-spread uncertainty, per-view votes, and a defer-to-human verdict
behind a small ``fit`` / ``evaluate`` / ``predict`` Trainer.

Train-time sibling of ViewConsensus, which wraps already-trained
classifiers at inference time with the same consensus semantics.
"""

from .core import ConsensusResult, ViewVote, consensus_batch, consensus_from_probabilities
from .trainer import (
    ConsensusTrainer,
    EpochRecord,
    EvalResult,
    History,
    Prediction,
    ThresholdFit,
    Vote,
    plain_trainer,
)
from .views import (
    View,
    brightness_view,
    contrast_view,
    default_views,
    horizontal_flip_view,
    identity_view,
    noise_view,
    rotate_view,
    shift_view,
)

__version__ = "0.1.0"

__all__ = [
    "ConsensusResult",
    "ConsensusTrainer",
    "EpochRecord",
    "EvalResult",
    "History",
    "Prediction",
    "ThresholdFit",
    "View",
    "ViewVote",
    "Vote",
    "brightness_view",
    "consensus_batch",
    "consensus_from_probabilities",
    "contrast_view",
    "default_views",
    "horizontal_flip_view",
    "identity_view",
    "noise_view",
    "plain_trainer",
    "rotate_view",
    "shift_view",
]
