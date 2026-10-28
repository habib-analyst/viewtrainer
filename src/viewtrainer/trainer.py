"""ConsensusTrainer: train any PyTorch classifier so its views must agree.

The training objective has two parts, computed over a panel of augmented
views of each input:

* a task loss (cross-entropy against the label), and
* a consensus loss: the mean KL divergence between each view's
  probability vector and the panel consensus (the mean probability,
  detached). By default it is *masked* to samples whose consensus
  prediction is already correct — the same confidence-masking idea
  FixMatch uses — so the loss tightens agreement where the panel is
  right instead of forcing confident agreement on ambiguous samples,
  which would destroy the spread signal the defer rule depends on.

At predict time the *same* panel votes again. The consensus probability,
the spread of the winning class across views (the uncertainty), every
view's vote, and a defer-to-human verdict are returned as one object.
The defer rule is a threshold on the view-spread uncertainty; the
threshold can be fitted on held-out data to a target accepted accuracy
(:meth:`ConsensusTrainer.fit_defer_threshold`). That operation is
defer-threshold fitting, not a calibration or coverage guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from .core import consensus_from_probabilities
from .views import View, default_views, identity_view

Loader = Sequence[Tuple[Tensor, Tensor]]


@dataclass
class Vote:
    """One view's vote inside a :class:`Prediction`."""

    view: str
    label_index: int
    probability: float


@dataclass
class Prediction:
    """One sample's consensus prediction."""

    label_index: int
    probability: float  # consensus probability of the winning class
    uncertainty: float  # winning-class spread across views
    verdict: str  # "confident" or "unsure"
    should_defer: bool
    agreement: float  # fraction of views voting for the winning class
    probabilities: np.ndarray  # [C] consensus probabilities
    class_uncertainty: np.ndarray  # [C] per-class spread
    votes: List[Vote] = field(default_factory=list)
    label: Optional[str] = None

    def explain(self) -> str:
        lines = [
            f"prediction: {self.label if self.label is not None else self.label_index} "
            f"({self.probability:.1%} consensus, +/-{self.uncertainty:.1%} across views)",
            f"agreement: {self.agreement:.0%} of views, verdict: {self.verdict}"
            + (" — defer to a human" if self.should_defer else ""),
            "votes:",
        ]
        for vote in self.votes:
            lines.append(
                f"  {vote.view:<18} -> class {vote.label_index} "
                f"({vote.probability:.1%})"
            )
        return "\n".join(lines)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    task_loss: float
    consensus_loss: float
    val_accuracy: Optional[float] = None


@dataclass
class History:
    """Per-epoch training history returned by :meth:`ConsensusTrainer.fit`."""

    epochs: List[EpochRecord] = field(default_factory=list)

    @property
    def train_loss(self) -> List[float]:
        return [e.train_loss for e in self.epochs]


@dataclass
class EvalResult:
    """Evaluation of the consensus predictor on a labeled loader."""

    loss: float  # cross-entropy of the consensus probabilities
    accuracy: float
    mean_uncertainty: float
    defer_rate: float  # fraction deferred at the current threshold
    accepted_accuracy: float  # accuracy among non-deferred samples
    aurc: float  # area under the risk-coverage curve (lower is better)
    risk_coverage: List[Tuple[float, float]]  # (coverage, risk) points

    def summary(self) -> str:
        return (
            f"accuracy {self.accuracy:.3f} | loss {self.loss:.4f} | "
            f"mean uncertainty {self.mean_uncertainty:.4f} | "
            f"defer rate {self.defer_rate:.1%} -> accepted accuracy "
            f"{self.accepted_accuracy:.3f} | AURC {self.aurc:.4f}"
        )


@dataclass
class ThresholdFit:
    """Result of fitting the defer threshold on held-out data."""

    threshold: float
    target_accepted_accuracy: float
    achieved_accepted_accuracy: float
    defer_rate: float
    met: bool  # whether the target was achieved on the fitting data


class ConsensusTrainer:
    """Wrap an ``nn.Module`` classifier with consensus training.

    Parameters
    ----------
    model:
        A PyTorch module mapping a batched tensor ``(N, C, H, W)`` to
        class logits ``(N, num_classes)``.
    labels:
        Optional class names, used for ``Prediction.label``.
    views:
        The view panel. Defaults to :func:`viewtrainer.default_views`.
        Views must be label-preserving for your task.
    consensus_weight:
        Weight of the consensus loss. ``0`` disables it, which reduces
        training to plain per-view (or consensus) cross-entropy — useful
        as an ablation baseline. The default is a deliberately gentle
        ``0.1``: in the measured demo sweep (REPORT.md), a weight of
        1.0 traded away substantial accuracy on a small task, while
        0.1 kept the panel's gains and tightened view agreement on
        correct predictions. Treat it as a dial, not a constant.
    consensus_mask:
        ``"correct"`` (default) applies the consensus loss only to
        samples whose consensus prediction matches the label;
        ``"none"`` applies it to every sample.
    task_loss:
        ``"per_view"`` averages cross-entropy over views (default);
        ``"consensus"`` applies cross-entropy to the mean probability.
    optimizer:
        ``None`` builds Adam over the model parameters with ``lr``;
        otherwise pass a configured ``torch.optim.Optimizer``.
    defer_threshold:
        Defer when the winning class's view spread exceeds this value.
        Refit it on held-out data with :meth:`fit_defer_threshold`.
    device:
        ``None`` selects CUDA when available, else CPU.
    seed:
        Seed for torch and NumPy at construction time.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        labels: Optional[Sequence[str]] = None,
        views: Optional[Sequence[Union[View, Callable[[Tensor], Tensor]]]] = None,
        consensus_weight: float = 0.1,
        consensus_mask: str = "correct",
        task_loss: str = "per_view",
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        defer_threshold: float = 0.10,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 0,
    ) -> None:
        if consensus_weight < 0:
            raise ValueError(f"consensus_weight must be >= 0, got {consensus_weight}")
        if consensus_mask not in ("correct", "none"):
            raise ValueError(
                f"consensus_mask must be 'correct' or 'none', got {consensus_mask!r}"
            )
        if task_loss not in ("per_view", "consensus"):
            raise ValueError(
                f"task_loss must be 'per_view' or 'consensus', got {task_loss!r}"
            )
        if not 0.0 <= defer_threshold <= 1.0:
            raise ValueError(f"defer_threshold must be in [0, 1], got {defer_threshold}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.labels = list(labels) if labels is not None else None
        if self.labels is not None and len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        raw_views = list(views) if views is not None else default_views()
        if len(raw_views) < 1:
            raise ValueError("at least one view is required")
        self.views: List[View] = [
            v if isinstance(v, View) else View(getattr(v, "__name__", "view"), v)
            for v in raw_views
        ]
        self.consensus_weight = float(consensus_weight)
        self.consensus_mask = consensus_mask
        self.task_loss = task_loss
        self.defer_threshold = float(defer_threshold)
        self.seed = seed
        if optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if params:
                optimizer = torch.optim.Adam(
                    params, lr=lr, weight_decay=weight_decay
                )
            # A parameter-free (or fully frozen) model gets no optimizer:
            # predict/evaluate still work, and fit() explains itself.
        self.optimizer = optimizer
        self.history_: Optional[History] = None
        self.num_classes_: Optional[int] = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _view_logits(self, x: Tensor) -> Tensor:
        """Forward ``x`` through every view; return logits ``[N, V, C]``."""
        return torch.stack([self.model(view(x)) for view in self.views], dim=1)

    def _check_labels(self, n_classes: int) -> None:
        if self.num_classes_ is None:
            self.num_classes_ = n_classes
            if self.labels is not None and len(self.labels) != n_classes:
                raise ValueError(
                    f"labels has {len(self.labels)} entries but the model "
                    f"outputs {n_classes} classes"
                )
        elif n_classes != self.num_classes_:
            raise ValueError(
                f"model output width changed from {self.num_classes_} to {n_classes}"
            )

    def _collect(self, loader: Loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (consensus probs [N, C], uncertainties [N], labels [N])."""
        self.model.eval()
        probs_out, unc_out, labels_out = [], [], []
        with torch.no_grad():
            for x, y in loader:
                view_probs = torch.softmax(self._view_logits(x.to(self.device)), dim=-1)
                arr = view_probs.cpu().numpy()
                for i in range(arr.shape[0]):
                    result = consensus_from_probabilities(arr[i])
                    probs_out.append(result.probabilities)
                    unc_out.append(result.uncertainty)
                labels_out.append(np.asarray(y))
        return (
            np.stack(probs_out),
            np.asarray(unc_out, dtype=np.float64),
            np.concatenate(labels_out),
        )

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Loader,
        *,
        val_loader: Optional[Loader] = None,
        epochs: int = 10,
    ) -> History:
        """Train with the consensus objective for ``epochs`` passes.

        Returns a :class:`History` with per-epoch total, task, and
        consensus losses, plus validation consensus accuracy when
        ``val_loader`` is given.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if self.optimizer is None:
            raise ValueError(
                "this trainer has no optimizer because the model has no "
                "trainable parameters; fit() needs a trainable model "
                "(predict/evaluate still work)"
            )
        history = History()
        for epoch in range(epochs):
            self.model.train()
            total = task_total = cons_total = 0.0
            n_batches = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self._view_logits(x)  # [N, V, C]
                self._check_labels(logits.shape[-1])
                probs = torch.softmax(logits, dim=-1)
                consensus = probs.mean(dim=1)  # [N, C]
                if self.task_loss == "per_view":
                    task = torch.stack(
                        [torch.nn.functional.cross_entropy(logits[:, v], y)
                         for v in range(logits.shape[1])]
                    ).mean()
                else:
                    task = torch.nn.functional.nll_loss(
                        torch.log(consensus.clamp_min(1e-12)), y
                    )
                log_probs = torch.log_softmax(logits, dim=-1)  # [N, V, C]
                kl_per_view = torch.nn.functional.kl_div(
                    log_probs,
                    consensus.detach().unsqueeze(1).expand_as(log_probs),
                    reduction="none",
                ).sum(dim=-1)  # [N, V]
                kl_per_sample = kl_per_view.mean(dim=1)  # [N]
                if self.consensus_mask == "correct":
                    mask = (
                        consensus.detach().argmax(dim=1) == y
                    ).to(kl_per_sample.dtype)
                    denom = mask.sum().clamp_min(1.0)
                    cons = (kl_per_sample * mask).sum() / denom
                else:
                    cons = kl_per_sample.mean()
                loss = task + self.consensus_weight * cons
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += float(loss.item())
                task_total += float(task.item())
                cons_total += float(cons.item())
                n_batches += 1
            if n_batches == 0:
                raise ValueError("train_loader produced no batches")
            record = EpochRecord(
                epoch=epoch,
                train_loss=total / n_batches,
                task_loss=task_total / n_batches,
                consensus_loss=cons_total / n_batches,
            )
            if val_loader is not None:
                record.val_accuracy = self.evaluate(val_loader).accuracy
            history.epochs.append(record)
        self.history_ = history
        return history

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def predict_view_probabilities(self, x: Union[Tensor, np.ndarray]) -> np.ndarray:
        """Per-view probabilities for ``x`` as an array ``[N, V, C]``.

        This is the same panel output ViewConsensus consumes, so a
        ViewTrainer model can also be wrapped by a ViewConsensus panel.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        x = x.to(self.device)
        single = x.ndim == 3
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(
                f"predict expects a (C, H, W) sample or (N, C, H, W) batch, "
                f"got shape {tuple(x.shape)}"
            )
        self.model.eval()
        with torch.no_grad():
            view_probs = torch.softmax(self._view_logits(x), dim=-1)
        self._check_labels(view_probs.shape[-1])
        return view_probs.cpu().numpy()

    def predict(self, x: Union[Tensor, np.ndarray]) -> Union[Prediction, List[Prediction]]:
        """Consensus prediction for one sample ``(C, H, W)`` or a batch.

        A single sample returns one :class:`Prediction`; a batch returns
        a list of them.
        """
        single = (isinstance(x, np.ndarray) and x.ndim == 3) or (
            isinstance(x, Tensor) and x.ndim == 3
        )
        view_probs = self.predict_view_probabilities(x)  # [N, V, C]
        predictions: List[Prediction] = []
        for i in range(view_probs.shape[0]):
            result = consensus_from_probabilities(view_probs[i])
            should_defer = result.uncertainty > self.defer_threshold
            votes = [
                Vote(
                    view=self.views[v].name,
                    label_index=result.votes[v].label_index,
                    probability=result.votes[v].probability,
                )
                for v in range(len(self.views))
            ]
            predictions.append(
                Prediction(
                    label_index=result.label_index,
                    probability=result.probability,
                    uncertainty=result.uncertainty,
                    verdict="unsure" if should_defer else "confident",
                    should_defer=should_defer,
                    agreement=result.agreement,
                    probabilities=result.probabilities,
                    class_uncertainty=result.class_uncertainty,
                    votes=votes,
                    label=(
                        self.labels[result.label_index]
                        if self.labels is not None
                        else None
                    ),
                )
            )
        return predictions[0] if single else predictions

    # ------------------------------------------------------------------
    # evaluation and defer threshold
    # ------------------------------------------------------------------
    def evaluate(self, loader: Loader) -> EvalResult:
        """Accuracy, defer behavior, and risk-coverage on a labeled loader."""
        probs, uncertainties, labels = self._collect(loader)
        self._check_labels(probs.shape[1])
        predicted = probs.argmax(axis=1)
        correct = predicted == labels
        accuracy = float(correct.mean())
        eps = 1e-12
        loss = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())
        accepted = uncertainties <= self.defer_threshold
        defer_rate = float(1.0 - accepted.mean())
        accepted_accuracy = (
            float(correct[accepted].mean()) if accepted.any() else float("nan")
        )
        # Risk-coverage: accept the most certain samples first.
        order = np.argsort(uncertainties)
        sorted_correct = correct[order]
        cumulative_errors = np.cumsum(~sorted_correct)
        counts = np.arange(1, len(labels) + 1)
        risks = cumulative_errors / counts
        aurc = float(risks.mean())
        points = []
        for coverage in np.linspace(0.1, 1.0, 10):
            k = max(1, int(round(coverage * len(labels))))
            points.append((float(k / len(labels)), float(risks[k - 1])))
        return EvalResult(
            loss=loss,
            accuracy=accuracy,
            mean_uncertainty=float(uncertainties.mean()),
            defer_rate=defer_rate,
            accepted_accuracy=accepted_accuracy,
            aurc=aurc,
            risk_coverage=points,
        )

    def fit_defer_threshold(
        self, loader: Loader, *, target_accepted_accuracy: float = 0.98
    ) -> ThresholdFit:
        """Fit the defer threshold on held-out data to a target accuracy.

        Picks the threshold with the lowest defer rate whose accepted
        accuracy on ``loader`` reaches ``target_accepted_accuracy``. If
        no threshold reaches it, the best accepted accuracy found is
        used and the returned fit reports ``met=False`` honestly.
        Sets ``self.defer_threshold`` to the fitted value.
        """
        if not 0.0 < target_accepted_accuracy <= 1.0:
            raise ValueError(
                f"target_accepted_accuracy must be in (0, 1], "
                f"got {target_accepted_accuracy}"
            )
        probs, uncertainties, labels = self._collect(loader)
        correct = probs.argmax(axis=1) == labels
        candidates = np.unique(np.concatenate([[0.0], uncertainties]))
        best_met: Optional[ThresholdFit] = None
        best_any: Optional[ThresholdFit] = None
        for threshold in candidates:
            accepted = uncertainties <= threshold
            if not accepted.any():
                continue
            acc = float(correct[accepted].mean())
            fit = ThresholdFit(
                threshold=float(threshold),
                target_accepted_accuracy=target_accepted_accuracy,
                achieved_accepted_accuracy=acc,
                defer_rate=float(1.0 - accepted.mean()),
                met=acc >= target_accepted_accuracy,
            )
            if best_any is None or acc > best_any.achieved_accepted_accuracy:
                best_any = fit
            if fit.met and (best_met is None or fit.defer_rate < best_met.defer_rate):
                best_met = fit
        chosen = best_met if best_met is not None else best_any
        if chosen is None:
            raise ValueError("loader produced no samples")
        self.defer_threshold = chosen.threshold
        return chosen

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> Path:
        """Save model weights plus the trainer configuration."""
        path = Path(path)
        payload = {
            "model": self.model.state_dict(),
            "trainer": {
                "labels": self.labels,
                "consensus_weight": self.consensus_weight,
                "consensus_mask": self.consensus_mask,
                "task_loss": self.task_loss,
                "defer_threshold": self.defer_threshold,
                "num_classes": self.num_classes_,
                "view_names": [v.name for v in self.views],
            },
        }
        torch.save(payload, path)
        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        model: nn.Module,
        *,
        views: Optional[Sequence[Union[View, Callable[[Tensor], Tensor]]]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "ConsensusTrainer":
        """Load weights saved by :meth:`save` into a fresh ``model``.

        View callables are not serialized; pass ``views`` again (it
        defaults to the default panel). If the saved view names do not
        match the provided panel, a ``ValueError`` is raised rather
        than silently predicting with a different panel than trained.
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = payload["trainer"]
        trainer = cls(
            model,
            labels=config["labels"],
            views=views,
            consensus_weight=config["consensus_weight"],
            consensus_mask=config.get("consensus_mask", "correct"),
            task_loss=config["task_loss"],
            defer_threshold=config["defer_threshold"],
            device=device,
        )
        saved_names = config["view_names"]
        current_names = [v.name for v in trainer.views]
        if saved_names != current_names:
            raise ValueError(
                f"saved model was trained with views {saved_names}, but the "
                f"provided panel is {current_names}; pass the training panel "
                f"to load()"
            )
        trainer.model.load_state_dict(payload["model"])
        trainer.num_classes_ = config["num_classes"]
        return trainer


def plain_trainer(model: nn.Module, **kwargs) -> ConsensusTrainer:
    """Ablation helper: a trainer with the consensus machinery switched off.

    Uses a single identity view and zero consensus weight, so ``fit``
    reduces to plain cross-entropy training and ``predict`` reports zero
    view-spread uncertainty. Useful as the honest baseline in examples
    and experiments.
    """
    kwargs.pop("views", None)
    kwargs.pop("consensus_weight", None)
    return ConsensusTrainer(
        model, views=[identity_view()], consensus_weight=0.0, **kwargs
    )
