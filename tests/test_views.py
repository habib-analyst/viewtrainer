import pytest
import torch

from viewtrainer import (
    brightness_view,
    contrast_view,
    default_views,
    horizontal_flip_view,
    noise_view,
    rotate_view,
    shift_view,
)


def test_default_views_preserve_shape():
    x = torch.rand(2, 1, 8, 8)
    views = default_views()
    assert len(views) == 6
    assert views[0].name == "identity"
    for view in views:
        y = view(x)
        assert y.shape == x.shape, view.name


def test_horizontal_flip_is_exact():
    x = torch.zeros(1, 1, 2, 3)
    x[0, 0, 0, 0] = 1.0
    y = horizontal_flip_view()(x)
    assert y[0, 0, 0, 2] == 1.0
    assert y.sum() == 1.0


def test_shift_matches_roll():
    x = torch.rand(1, 1, 5, 5)
    y = shift_view(2, -1)(x)
    assert torch.equal(y, torch.roll(x, shifts=(-1, 2), dims=(2, 3)))


def test_noise_view_is_deterministic():
    x = torch.rand(2, 1, 6, 6)
    view = noise_view(0.05, seed=7)
    first, second = view(x), view(x)
    assert torch.equal(first, second)
    assert not torch.equal(first, x)


def test_contrast_of_constant_image_is_identity():
    x = torch.full((1, 1, 4, 4), 0.4)
    assert torch.allclose(contrast_view(1.5)(x), x)


def test_brightness_scales():
    x = torch.full((1, 1, 2, 2), 0.5)
    assert torch.allclose(brightness_view(1.2)(x), torch.full_like(x, 0.6))


def test_rotate_preserves_shape():
    x = torch.rand(1, 3, 10, 10)
    assert rotate_view(8.0)(x).shape == x.shape


def test_views_reject_unbatched_input():
    with pytest.raises(ValueError):
        default_views()[1](torch.rand(1, 8, 8))
