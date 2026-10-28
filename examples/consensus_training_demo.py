"""Consensus training vs plain training on a synthetic shape task.

A small CNN classifies 24x24 grayscale images into four shapes (disc,
square, triangle, ring) drawn at random positions and scales. About a
quarter of the samples are rendered *faint* — barely above the noisy
background — so they sit near the decision boundary by construction.

Three identical CNNs are trained from the same initial weights:

* **plain** — ordinary cross-entropy training (``plain_trainer``);
* **panel only** — the default six-view panel with the consensus loss
  switched off (``consensus_weight=0``): augmentation training alone;
* **consensus** — the same panel with the consensus loss switched on
  at the library's default gentle weight (``consensus_weight=0.1``).

The three-arm design isolates what the consensus loss itself adds,
separately from the well-known benefit of training on augmentations.

We then report, on the held-out test split: raw accuracy of all three
models, AURC and error separation for the panel models, and the
consensus model's accepted accuracy after deferring at a threshold
fitted on the validation split to target 95% accepted accuracy.

This is a mechanics demo on synthetic data with fixed seeds, not a
benchmark. The printed numbers are the real measured output of this
script; the same JSON is written to ``examples/output/demo_summary.json``.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from viewtrainer import ConsensusTrainer, consensus_from_probabilities, plain_trainer

LABELS = ["disc", "square", "triangle", "ring"]
SIZE = 24
SEED = 42


def draw_shape(rng: np.random.Generator, label: int, faint: bool) -> np.ndarray:
    img = 0.12 + 0.03 * rng.standard_normal((SIZE, SIZE))
    cx, cy = rng.uniform(8, 16, size=2)
    r = rng.uniform(4.5, 6.5)
    amp = rng.uniform(0.20, 0.32) if faint else rng.uniform(0.6, 1.0)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    if label == 0:  # disc
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    elif label == 1:  # square
        mask = (np.abs(xx - cx) <= r * 0.8) & (np.abs(yy - cy) <= r * 0.8)
    elif label == 2:  # triangle
        ax, ay = cx, cy - r
        bx, by = cx - 0.9 * r, cy + 0.8 * r
        cx2, cy2 = cx + 0.9 * r, cy + 0.8 * r

        def sign(px, py, qx, qy, rx, ry):
            return (px - rx) * (qy - ry) - (qx - rx) * (py - ry)

        d1 = sign(xx, yy, ax, ay, bx, by)
        d2 = sign(xx, yy, bx, by, cx2, cy2)
        d3 = sign(xx, yy, cx2, cy2, ax, ay)
        mask = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
    else:  # ring
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (d2 <= r**2) & (d2 >= (0.55 * r) ** 2)
    img[mask] = amp + 0.03 * rng.standard_normal(int(mask.sum()))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_split(n: int, faint_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n):
        label = i % len(LABELS)
        faint = rng.random() < faint_fraction
        xs.append(draw_shape(rng, label, faint))
        ys.append(label)
    x = torch.from_numpy(np.stack(xs)).unsqueeze(1)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def loader(x, y, batch=64, shuffle=False):
    # A fixed-seed generator makes the batch order identical for every
    # arm, so the three trainings are a paired comparison and the demo
    # does not depend on global RNG state left over from earlier arms.
    generator = torch.Generator().manual_seed(1000) if shuffle else None
    return DataLoader(
        TensorDataset(x, y), batch_size=batch, shuffle=shuffle, generator=generator
    )


class TinyCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.features(x)


def main() -> None:
    torch.manual_seed(SEED)
    x_train, y_train = make_split(1200, 0.25, seed=1)
    x_val, y_val = make_split(300, 0.25, seed=2)
    x_test, y_test = make_split(400, 0.30, seed=3)
    train_loader = loader(x_train, y_train, shuffle=True)
    val_loader = loader(x_val, y_val)
    test_loader = loader(x_test, y_test)

    torch.manual_seed(123)
    base_model = TinyCNN()
    plain = plain_trainer(copy.deepcopy(base_model), labels=LABELS, lr=1.5e-3, seed=SEED)
    panel_only = ConsensusTrainer(
        copy.deepcopy(base_model), labels=LABELS, lr=1.5e-3, seed=SEED,
        consensus_weight=0.0,
    )
    consensus = ConsensusTrainer(
        copy.deepcopy(base_model), labels=LABELS, lr=1.5e-3, seed=SEED
    )

    started = time.time()
    print("training plain baseline (identity view, no consensus loss) ...")
    plain.fit(train_loader, epochs=8)
    print("training panel-only ablation (6-view panel, consensus_weight=0) ...")
    panel_only.fit(train_loader, epochs=8)
    print("training consensus model (6-view panel, consensus_weight=0.1) ...")
    history = consensus.fit(train_loader, val_loader=val_loader, epochs=8)
    for record in history.epochs:
        print(
            f"  epoch {record.epoch}: loss {record.train_loss:.4f} "
            f"(task {record.task_loss:.4f} + consensus {record.consensus_loss:.4f}) "
            f"val accuracy {record.val_accuracy:.3f}"
        )

    plain_eval = plain.evaluate(test_loader)
    panel_eval = panel_only.evaluate(test_loader)
    raw_eval = consensus.evaluate(test_loader)

    fit = consensus.fit_defer_threshold(val_loader, target_accepted_accuracy=0.95)
    defer_eval = consensus.evaluate(test_loader)

    def risk_at_coverage(eval_result, coverage=0.8):
        return min(eval_result.risk_coverage, key=lambda p: abs(p[0] - coverage))[1]

    panel_risk80 = risk_at_coverage(panel_eval)
    consensus_risk80 = risk_at_coverage(raw_eval)

    # Does view-spread separate errors from correct predictions?
    def spread_separation(trainer):
        view_probs = trainer.predict_view_probabilities(x_test)
        uncertainties, correct = [], []
        for i in range(view_probs.shape[0]):
            result = consensus_from_probabilities(view_probs[i])
            uncertainties.append(result.uncertainty)
            correct.append(result.label_index == int(y_test[i]))
        uncertainties = np.array(uncertainties)
        correct = np.array(correct)
        return float(uncertainties[correct].mean()), float(uncertainties[~correct].mean())

    panel_unc_correct, panel_unc_wrong = spread_separation(panel_only)
    unc_correct, unc_wrong = spread_separation(consensus)

    summary = {
        "demo": "synthetic shapes (disc/square/triangle/ring), faint hard subset",
        "seed": SEED,
        "train_seconds": round(time.time() - started, 1),
        "plain": {"test_accuracy": round(plain_eval.accuracy, 4)},
        "panel_only": {
            "test_accuracy": round(panel_eval.accuracy, 4),
            "aurc": round(panel_eval.aurc, 4),
            "risk_at_80pct_coverage": round(panel_risk80, 4),
            "mean_uncertainty_correct": round(panel_unc_correct, 4),
            "mean_uncertainty_incorrect": round(panel_unc_wrong, 4),
        },
        "consensus": {
            "test_accuracy": round(raw_eval.accuracy, 4),
            "aurc": round(raw_eval.aurc, 4),
            "risk_at_80pct_coverage": round(consensus_risk80, 4),
            "mean_uncertainty_correct": round(unc_correct, 4),
            "mean_uncertainty_incorrect": round(unc_wrong, 4),
        },
        "defer": {
            "val_target_accepted_accuracy": 0.95,
            "val_target_met": fit.met,
            "fitted_threshold": round(fit.threshold, 4),
            "test_defer_rate": round(defer_eval.defer_rate, 4),
            "test_accepted_accuracy": round(defer_eval.accepted_accuracy, 4),
        },
    }

    print("\n=== measured results (synthetic demo, seed 42) ===")
    print(f"plain test accuracy:            {plain_eval.accuracy:.3f}")
    print(f"panel-only test accuracy:       {panel_eval.accuracy:.3f}")
    print(f"panel-only AURC:                {panel_eval.aurc:.4f} (lower is better)")
    print(f"panel-only view-spread, correct/incorrect: "
          f"{panel_unc_correct:.4f} / {panel_unc_wrong:.4f}")
    print(f"panel-only risk at 80% coverage: {panel_risk80:.4f}")
    print(f"consensus test accuracy:        {raw_eval.accuracy:.3f}")
    print(f"consensus AURC:                 {raw_eval.aurc:.4f} (lower is better)")
    print(f"consensus risk at 80% coverage: {consensus_risk80:.4f}")
    print(f"consensus view-spread, correct/incorrect:  "
          f"{unc_correct:.4f} / {unc_wrong:.4f}")
    print(
        f"defer threshold fitted on val:  {fit.threshold:.4f} "
        f"(target 0.95 accepted accuracy, met on val: {fit.met})"
    )
    print(f"test defer rate:                {defer_eval.defer_rate:.1%}")
    print(f"test accepted accuracy:         {defer_eval.accepted_accuracy:.3f}")

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_dir / 'demo_summary.json'}")


if __name__ == "__main__":
    main()
