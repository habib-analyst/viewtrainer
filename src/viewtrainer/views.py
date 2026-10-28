"""Augmented views for consensus training and prediction.

A *view* is a named, label-preserving transformation of a batched float
image tensor ``(N, C, H, W)``. The same panel of views is used in two
places, and that reuse is the point of ViewTrainer:

* during :meth:`ConsensusTrainer.fit`, every view of a training input is
  forwarded through the model and the views are trained to agree;
* during :meth:`ConsensusTrainer.predict`, the views vote again and their
  spread becomes the uncertainty signal.

The default panel targets float image tensors roughly in ``[0, 1]`` and
uses only plain PyTorch ops, so it adds no dependency beyond ``torch``.
Views must be label-preserving for the task at hand: do not include a
horizontal flip in the panel if flipping changes the label (for example
for left/right or text-direction classes). Supply your own callables if
the default panel does not fit your data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

ViewFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class View:
    """A named label-preserving transformation of a batched tensor."""

    name: str
    fn: ViewFn

    def __call__(self, x: Tensor) -> Tensor:
        return self.fn(x)


def _check_batched(x: Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(
            f"views expect a batched image tensor of shape (N, C, H, W), "
            f"got shape {tuple(x.shape)}"
        )


def identity_view() -> View:
    """The unmodified input. Always the first view of a panel."""

    def fn(x: Tensor) -> Tensor:
        _check_batched(x)
        return x

    return View("identity", fn)


def horizontal_flip_view() -> View:
    """Mirror the image left-right."""

    def fn(x: Tensor) -> Tensor:
        _check_batched(x)
        return torch.flip(x, dims=[3])

    return View("hflip", fn)


def shift_view(dx: int = 2, dy: int = -1) -> View:
    """Translate by ``(dx, dy)`` pixels with wrap-around (``torch.roll``)."""

    def fn(x: Tensor) -> Tensor:
        _check_batched(x)
        return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

    return View(f"shift({dx:+d},{dy:+d})", fn)


def brightness_view(factor: float = 1.15) -> View:
    """Scale intensities by ``factor`` (no clamping)."""
    if factor <= 0:
        raise ValueError(f"brightness factor must be positive, got {factor}")

    def fn(x: Tensor) -> Tensor:
        _check_batched(x)
        return x * factor

    return View(f"brightness({factor:g})", fn)


def contrast_view(factor: float = 1.2) -> View:
    """Scale the distance from each image's per-channel mean by ``factor``."""
    if factor <= 0:
        raise ValueError(f"contrast factor must be positive, got {factor}")

    def fn(x: Tensor) -> Tensor:
        _check_batched(x)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return (x - mean) * factor + mean

    return View(f"contrast({factor:g})", fn)


def rotate_view(degrees: float = 8.0) -> View:
    """Rotate by ``degrees`` around the centre, zero padding outside."""
    if degrees == 0:
        raise ValueError("rotation angle must be non-zero")

    def fn(x: Tensor) -> Tensor:
        _check_batched(x)
        angle = degrees * 3.141592653589793 / 180.0
        cos_a, sin_a = float(torch.cos(torch.tensor(angle))), float(
            torch.sin(torch.tensor(angle))
        )
        theta = x.new_tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]
        ).unsqueeze(0)
        theta = theta.expand(x.shape[0], -1, -1)
        grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
        return torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    return View(f"rotate({degrees:g})", fn)


def noise_view(sigma: float = 0.05, seed: int = 1234) -> View:
    """Additive Gaussian noise with a fixed seed.

    The generator is re-seeded on every call, so this view is
    deterministic: the same input always produces the same noisy view,
    in training and at predict time.
    """
    if sigma < 0:
        raise ValueError(f"noise sigma must be non-negative, got {sigma}")

    def fn(x: Tensor) -> Tensor:
        _check_batched(x)
        generator = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return x + sigma * noise

    return View(f"noise({sigma:g})", fn)


def default_views() -> list[View]:
    """The default six-view panel.

    identity, horizontal flip, small shift, brightness up, contrast up,
    small rotation. All are label-preserving for typical shape/object
    image classification; check them against your task before use.
    """
    return [
        identity_view(),
        horizontal_flip_view(),
        shift_view(2, -1),
        brightness_view(1.15),
        contrast_view(1.2),
        rotate_view(8.0),
    ]
