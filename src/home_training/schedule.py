"""Learning-rate schedule: linear warmup into cosine decay.

Warmup exists because AdamW's second-moment estimate is meaningless for the
first few hundred steps; stepping at full rate before it settles is a common
way to diverge in the first minute of a run that would otherwise have worked.
"""

from __future__ import annotations

import math

__all__ = ["cosine_with_warmup"]


def cosine_with_warmup(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Learning rate for ``step`` (0-indexed).

    Decays to ``base_lr * min_lr_ratio`` rather than to zero: a run that ends
    at exactly zero spends its final steps not learning, and the floor leaves
    room to extend training later without a discontinuity.
    """
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and total_steps > 0")
    min_lr = base_lr * min_lr_ratio

    if step < warmup_steps:
        # step + 1 so the very first step is non-zero.
        return base_lr * (step + 1) / max(1, warmup_steps)

    if step >= total_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
