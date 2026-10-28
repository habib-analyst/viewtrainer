import numpy as np
import pytest
import torch

from viewtrainer import consensus_batch, consensus_from_probabilities


def test_agreeing_views_have_zero_uncertainty():
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    result = consensus_from_probabilities(probs)
    assert result.label_index == 0
    assert result.probability == pytest.approx(0.9)
    assert result.uncertainty == 0.0
    assert result.agreement == 1.0
    assert len(result.votes) == 3
    assert all(v.label_index == 0 for v in result.votes)


def test_disagreement_spread_matches_population_std():
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    result = consensus_from_probabilities(probs)
    assert result.label_index == 0
    assert result.probability == pytest.approx(0.6333, abs=1e-3)
    expected = np.std(np.array([0.9, 0.1, 0.9]))  # population std
    assert result.uncertainty == pytest.approx(expected)
    assert result.agreement == pytest.approx(2 / 3)
    assert result.votes[1].label_index == 1


def test_torch_tensor_input_accepted():
    probs = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
    result = consensus_from_probabilities(probs)
    assert result.label_index == 1
    assert result.probabilities.sum() == pytest.approx(1.0)


def test_batch_matches_per_sample():
    rng = np.random.default_rng(0)
    raw = rng.random((5, 3, 4))
    arr = raw / raw.sum(axis=2, keepdims=True)
    batch = consensus_batch(arr)
    assert len(batch) == 5
    for i, result in enumerate(batch):
        single = consensus_from_probabilities(arr[i])
        assert result.label_index == single.label_index
        assert result.uncertainty == pytest.approx(single.uncertainty)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0.5, 0.6], [0.5, 0.5]]),  # row does not sum to 1
        np.array([[np.nan, 1.0], [0.5, 0.5]]),  # non-finite
        np.array([[1.1, -0.1], [0.5, 0.5]]),  # negative
        np.array([0.5, 0.5]),  # 1D
        np.zeros((0, 2)),  # no views
        np.array([[1.0], [1.0]]),  # one class
    ],
)
def test_validation_rejects_bad_probabilities(bad):
    with pytest.raises(ValueError):
        consensus_from_probabilities(bad)
