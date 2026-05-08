"""Tests for ``src.gpt2.utils``."""

from __future__ import annotations

import pandas as pd
import pytest
from transformers import GPT2Tokenizer

from src.gpt2.config import CONFIG
from src.gpt2.preprocess import load_data
from src.gpt2.utils import build_dataloaders, split_data


@pytest.fixture
def tokenizer():
    tok = GPT2Tokenizer.from_pretrained(CONFIG.model_name)
    tok.add_special_tokens(
        {
            "pad_token": CONFIG.pad_token,
            "bos_token": CONFIG.bos_token,
            "eos_token": CONFIG.eos_token,
        }
    )
    return tok


class TestUtils:
    def test_split_data(self, temp_reuters_headlines):
        df = load_data(file_path=temp_reuters_headlines)

        train_description, test_description = split_data(df, test_size=0.20, random_state=42)

        assert not isinstance(train_description, pd.DataFrame)
        assert not isinstance(test_description, pd.DataFrame)
        assert len(str(train_description)) > len(str(test_description))

    def test_build_dataloaders(self, temp_reuters_headlines, tokenizer):
        df = load_data(file_path=temp_reuters_headlines)

        train_description, test_description = split_data(df, test_size=0.20, random_state=42)

        train_loader, test_loader = build_dataloaders(
            train_description, test_description, tokenizer
        )

        assert "input_ids" in next(iter(train_loader))
        assert "attention_mask" in next(iter(train_loader))
        assert "input_ids" in next(iter(test_loader))
        assert "attention_mask" in next(iter(test_loader))
