# Sampling/schedule helpers for Chroma.
#
# Chroma uses rectified-flow sampling like FLUX, and reuses FLUX's noise generation, latent packing,
# and image-id helpers (see invokeai.backend.flux.sampling_utils). The schedule construction below is
# Chroma-specific (it adds a "sine" schedule and exposes the shift toggle directly).
#
# Portions adapted from https://github.com/lodestone-rock/flow

import math
from typing import Callable

import torch
from torch import Tensor


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def time_shift_inverse(mu: float, sigma: float, y: Tensor):
    term = torch.pow(math.exp(mu) * (1 - y) / y, 1 / sigma)
    t = 1 / (1 + term)
    return t


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    *,
    start: float = 1.0,
    end: float = 0.0,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """Build a rectified-flow timestep schedule for Chroma.

    If ``shift`` is True, the schedule is resolution-dependent (more steps at high noise), matching
    the FLUX.1 [dev] behaviour. If False, the steps are placed linearly.
    """
    if shift:
        # estimate mu based on a linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        bounds = time_shift_inverse(mu, 1.0, torch.tensor([start, end]))
        timesteps = torch.linspace(bounds[0].item(), bounds[1].item(), num_steps + 1)
        timesteps = time_shift(mu, 1.0, timesteps)
    else:
        # extra step for zero
        timesteps = torch.linspace(start, end, num_steps + 1)

    return timesteps.tolist()


def sinelike(x, k=1.0, shift=1.0):
    y = torch.sin((x**shift * 2 - 1) * torch.pi / 2)
    return torch.copysign(torch.pow(torch.abs(y), k), y) / 2 + 0.5


def sinelike_inverse(y, k=1.0, shift=1.0):
    w = 2 * y - 1
    v = torch.copysign(torch.pow(torch.abs(w), 1.0 / k), w)
    u = ((2.0 / torch.pi) * torch.arcsin(v) + 1) / 2.0
    x = torch.pow(u, 1.0 / shift)
    return x


def get_schedule_sinelike(num_steps: int, *, start: float = 1.0, end: float = 0.0, k: float = 1.0, shift: float = 1.0):
    """A schedule that emphasizes both the start and end of the timeline (steps are scarce in the
    middle). With the default smoothness it approximates the shifted/sigmoid curve."""
    bounds = sinelike_inverse(torch.tensor([start, end]), k=k, shift=shift)
    return sinelike(torch.linspace(bounds[0].item(), bounds[1].item(), num_steps + 1), k, shift).tolist()
