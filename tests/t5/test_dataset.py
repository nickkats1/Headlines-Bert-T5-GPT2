"""Tests for ``src.t5.dataset``."""

from __future__ import annotations

import pytest
from transformers import T5Tokenizer

from src.t5.config import CONFIG
from src.t5.dataset import CustomDataset
from src.t5.preprocess import load_data


@pytest.fixture
def tokenizer():
    return T5Tokenizer.from_pretrained(CONFIG.model_name)


@pytest.fixture
def custom_dataset(temp_reuters_headlines, tokenizer):
    df = load_data(file_path=temp_reuters_headlines)
    return CustomDataset(
        dataframe=df,
        tokenizer=tokenizer,
        source_len=CONFIG.source_length,
        target_len=CONFIG.target_length,
        source_col="Description",
        target_col="Headlines",
    )


class TestCustomDataset:
    def test_attributes(self, custom_dataset):
        assert isinstance(custom_dataset, CustomDataset)
        assert custom_dataset.source_len == CONFIG.source_length
        assert custom_dataset.target_len == CONFIG.target_length
        assert hasattr(custom_dataset, "tokenizer")
        assert hasattr(custom_dataset, "data")
        assert hasattr(custom_dataset, "source_text")
        assert hasattr(custom_dataset, "target_text")

    def test_len(self, custom_dataset):
        assert len(custom_dataset) > 0

    def test_getitem_key(self, custom_dataset):
        sample = custom_dataset[0]

        assert sample["source_ids"].shape == (CONFIG.source_length,)
        assert sample["source_mask"].shape == (CONFIG.source_length,)
        assert sample["target_ids"].shape == (CONFIG.target_length,)

        assert "source_ids" in sample
        assert "source_mask" in sample
        assert "target_ids" in sample
