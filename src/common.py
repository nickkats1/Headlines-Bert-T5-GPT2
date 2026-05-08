"""Shared utilities used across BERT, GPT-2, and T5 pipelines."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU & CUDA) RNGs.

    Args:
        seed: Non-negative integer used to seed every RNG.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative; got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str | None = None) -> torch.device:
    """Resolve a torch device, falling back to CPU when CUDA is unavailable.

    Args:
        preferred: Optional preferred device string (``"cuda"``, ``"cuda:0"``,
            ``"cpu"``, ...). If ``None`` (default) CUDA is selected when
            available, otherwise CPU.

    Returns:
        ``torch.device`` instance suitable for ``.to(...)``.
    """
    if preferred is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(preferred)
