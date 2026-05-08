"""Tests for ``src.t5.metrics``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.t5.metrics import ROUGE_KEYS, compute_rouge, save_predictions


class TestComputeRouge:
    """Test ``compute_rouge``."""

    def test_empty_inputs_returns_zeros(self):
        """All ROUGE scores are 0.0 when no pairs are provided."""
        result = compute_rouge([], [])
        assert set(result.keys()) == set(ROUGE_KEYS)
        assert all(v == 0.0 for v in result.values())

    def test_perfect_match_returns_one(self):
        """Identical predictions and actuals produce perfect ROUGE-1/2/L."""
        preds = ["The quick brown fox jumps over the lazy dog"]
        actuals = ["The quick brown fox jumps over the lazy dog"]
        result = compute_rouge(preds, actuals)
        for key in ROUGE_KEYS:
            assert result[key] == pytest.approx(1.0)

    def test_keys_present(self):
        """All ROUGE keys are returned."""
        preds = ["something"]
        actuals = ["different text"]
        result = compute_rouge(preds, actuals)
        for key in ROUGE_KEYS:
            assert key in result
            assert 0.0 <= result[key] <= 1.0

    def test_length_mismatch_raises(self):
        """Mismatched-length inputs raise ValueError."""
        with pytest.raises(ValueError):
            compute_rouge(["a"], ["a", "b"])


class TestSavePredictions:
    """Test ``save_predictions``."""

    def test_writes_csv_with_expected_columns(self, tmp_path):
        """save_predictions writes a 2-column CSV in the expected order."""
        preds = ["pred one", "pred two"]
        actuals = ["actual one", "actual two"]
        out_path = tmp_path / "out.csv"

        result_path = save_predictions(preds, actuals, path=out_path)

        assert result_path == out_path
        assert out_path.is_file()

        df = pd.read_csv(out_path)
        assert list(df.columns) == ["Generated Text", "Actual Text"]
        assert df["Generated Text"].tolist() == preds
        assert df["Actual Text"].tolist() == actuals

    def test_creates_parent_dir(self, tmp_path):
        """Parent directories are created automatically."""
        out_path = tmp_path / "deep" / "nested" / "preds.csv"
        save_predictions(["x"], ["y"], path=out_path)
        assert out_path.is_file()

    def test_length_mismatch_raises(self, tmp_path):
        """Mismatched-length inputs raise ValueError."""
        with pytest.raises(ValueError):
            save_predictions(["a"], ["a", "b"], path=tmp_path / "out.csv")
