"""Tests for ``src.common``."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.common import resolve_device, seed_everything


class TestSeedEverything:
    """Test ``seed_everything``."""

    def test_reproducible_random(self):
        """Python ``random`` produces deterministic output after seeding."""
        seed_everything(42)
        a = [random.random() for _ in range(5)]
        seed_everything(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_reproducible_numpy(self):
        """NumPy RNG is deterministic after seeding."""
        seed_everything(42)
        a = np.random.rand(5)
        seed_everything(42)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_reproducible_torch(self):
        """PyTorch RNG is deterministic after seeding."""
        seed_everything(42)
        a = torch.rand(5)
        seed_everything(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_negative_seed_raises(self):
        """Negative seeds are rejected."""
        with pytest.raises(ValueError):
            seed_everything(-1)


class TestResolveDevice:
    """Test ``resolve_device``."""

    def test_default_returns_torch_device(self):
        """Default returns a valid torch.device."""
        device = resolve_device()
        assert isinstance(device, torch.device)
        assert device.type in {"cuda", "cpu"}

    def test_cpu_passthrough(self):
        """Explicit cpu request returns CPU."""
        assert resolve_device("cpu").type == "cpu"

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        """CUDA request falls back to CPU when CUDA is unavailable."""
        if torch.cuda.is_available():
            pytest.skip("CUDA available; cannot test fallback.")
        assert resolve_device("cuda").type == "cpu"
        assert resolve_device("cuda:0").type == "cpu"
