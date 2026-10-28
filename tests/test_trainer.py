import numpy as np
import pytest
import torch
from torch import nn

from viewtrainer import ConsensusTrainer, default_views, identity_view, plain_trainer


class TinyNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ConstantNet(nn.Module):
    """Always emits the same logits, so every view agrees exactly."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x):
        logits = torch.tensor([2.0, 0.0])[: self.bias.shape[0]]
        return logits.expand(x.shape[0], -1) + 0.0 * self.bias.sum()


class HalfNet(nn.Module):
    """Predicts by which horizontal half is brighter; a flip flips it."""

    def forward(self, x):
        left = x[..., :4].mean(dim=(1, 2, 3))
        right = x[..., 4:].mean(dim=(1, 2, 3))
        return torch.stack([left, right], dim=1) * 5.0


def make_loader(n=64, batch=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for i in range(n):
        label = i % 2
        img = 0.1 + 0.1 * torch.rand(1, 8, 8, generator=g)
        if label == 0:
            img[..., :4] += 0.7
        else:
            img[..., 4:] += 0.7
        xs.append(img)
        ys.append(label)
    x = torch.stack(xs)
    y = torch.tensor(ys)
    return [(x[i : i + batch], y[i : i + batch]) for i in range(0, n, batch)]


def test_fit_reduces_loss_and_records_val_accuracy():
    loader = make_loader()
    trainer = ConsensusTrainer(TinyNet(), lr=0.05, seed=0)
    history = trainer.fit(loader, val_loader=loader, epochs=5)
    assert len(history.epochs) == 5
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.epochs[-1].val_accuracy is not None
    assert history.epochs[-1].val_accuracy > 0.8
    assert history.epochs[0].consensus_loss >= 0.0


def test_predict_single_and_batch():
    loader = make_loader(n=8)
    trainer = ConsensusTrainer(
        TinyNet(), labels=["left", "right"], lr=0.05, seed=0
    )
    trainer.fit(loader, epochs=3)
    x = loader[0][0]
    single = trainer.predict(x[0])
    assert single.label in ("left", "right")
    assert len(single.votes) == len(default_views())
    assert single.probabilities.sum() == pytest.approx(1.0, abs=1e-5)
    assert single.verdict in ("confident", "unsure")
    assert "votes:" in single.explain()
    batch = trainer.predict(x)
    assert isinstance(batch, list) and len(batch) == x.shape[0]
    assert batch[0].label_index == single.label_index


def test_predict_view_probabilities_shape():
    trainer = ConsensusTrainer(TinyNet(), seed=0)
    arr = trainer.predict_view_probabilities(torch.rand(3, 1, 8, 8))
    assert arr.shape == (3, len(default_views()), 2)
    assert np.allclose(arr.sum(axis=2), 1.0, atol=1e-5)


def test_agreeing_model_has_zero_uncertainty_and_never_defers():
    trainer = ConsensusTrainer(ConstantNet(), defer_threshold=0.05, seed=0)
    pred = trainer.predict(torch.rand(1, 8, 8))
    assert pred.uncertainty == 0.0
    assert pred.agreement == 1.0
    assert pred.verdict == "confident"
    assert pred.should_defer is False


def test_disagreeing_views_trigger_defer():
    trainer = ConsensusTrainer(
        HalfNet(), defer_threshold=0.0, seed=0
    )
    img = torch.full((1, 8, 8), 0.1)
    img[..., :4] += 0.7  # identity says class 0, hflip view says class 1
    pred = trainer.predict(img)
    assert pred.uncertainty > 0.0
    assert pred.agreement < 1.0
    assert pred.should_defer is True
    assert pred.verdict == "unsure"


def test_evaluate_reports_risk_coverage():
    loader = make_loader()
    trainer = ConsensusTrainer(TinyNet(), lr=0.05, seed=0)
    trainer.fit(loader, epochs=4)
    result = trainer.evaluate(loader)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.aurc <= 1.0
    assert len(result.risk_coverage) == 10
    coverage, risk = result.risk_coverage[-1]
    assert coverage == pytest.approx(1.0)
    assert risk == pytest.approx(1.0 - result.accuracy)
    assert 0.0 <= result.defer_rate <= 1.0


def test_fit_defer_threshold_met_and_unmet():
    loader = make_loader()
    trainer = ConsensusTrainer(TinyNet(), lr=0.05, seed=0)
    trainer.fit(loader, epochs=5)
    fit = trainer.fit_defer_threshold(loader, target_accepted_accuracy=0.9)
    assert fit.threshold == trainer.defer_threshold
    assert 0.0 <= fit.defer_rate <= 1.0
    if fit.met:
        assert fit.achieved_accepted_accuracy >= 0.9

    wrong = ConsensusTrainer(ConstantNet(), seed=0)
    wrong_loader = [(torch.rand(8, 1, 8, 8), torch.ones(8, dtype=torch.long))]
    unmet = wrong.fit_defer_threshold(wrong_loader, target_accepted_accuracy=0.99)
    assert unmet.met is False  # constant model is always wrong here; honest report


def test_plain_trainer_ablation_has_zero_spread():
    loader = make_loader(n=16)
    trainer = plain_trainer(TinyNet(), lr=0.05, seed=0)
    trainer.fit(loader, epochs=2)
    pred = trainer.predict(loader[0][0][0])
    assert pred.uncertainty == 0.0
    assert len(pred.votes) == 1


def test_consensus_mask_none_variant_runs():
    loader = make_loader(n=16)
    trainer = ConsensusTrainer(
        TinyNet(), consensus_mask="none", lr=0.05, seed=0
    )
    history = trainer.fit(loader, epochs=2)
    assert len(history.epochs) == 2
    assert history.epochs[-1].consensus_loss >= 0.0


def test_consensus_task_loss_variant_runs():
    loader = make_loader(n=16)
    trainer = ConsensusTrainer(
        TinyNet(), task_loss="consensus", lr=0.05, seed=0
    )
    history = trainer.fit(loader, epochs=2)
    assert len(history.epochs) == 2


def test_save_load_round_trip_and_view_guard(tmp_path):
    loader = make_loader(n=16)
    trainer = ConsensusTrainer(TinyNet(), labels=["l", "r"], lr=0.05, seed=0)
    trainer.fit(loader, epochs=2)
    x = loader[0][0][0]
    before = trainer.predict(x)
    path = trainer.save(tmp_path / "model.pt")
    loaded = ConsensusTrainer.load(path, TinyNet())
    after = loaded.predict(x)
    assert after.label_index == before.label_index
    assert after.probability == pytest.approx(before.probability, abs=1e-6)
    with pytest.raises(ValueError):
        ConsensusTrainer.load(path, TinyNet(), views=[identity_view()])


def test_parameter_free_model_predicts_but_fit_explains_itself():
    trainer = ConsensusTrainer(HalfNet(), seed=0)
    pred = trainer.predict(torch.rand(1, 8, 8))
    assert pred.probabilities.sum() == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match="no trainable parameters"):
        trainer.fit(make_loader(n=8), epochs=1)


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        ConsensusTrainer(TinyNet(), consensus_weight=-0.5)
    with pytest.raises(ValueError):
        ConsensusTrainer(TinyNet(), consensus_mask="bogus")
    with pytest.raises(ValueError):
        ConsensusTrainer(TinyNet(), task_loss="bogus")
    with pytest.raises(ValueError):
        ConsensusTrainer(TinyNet(), views=[])
    with pytest.raises(ValueError):
        ConsensusTrainer(TinyNet(), defer_threshold=1.5)
    trainer = ConsensusTrainer(TinyNet())
    with pytest.raises(ValueError):
        trainer.fit(make_loader(n=8), epochs=0)
    with pytest.raises(ValueError):
        trainer.predict(torch.rand(8, 8))
