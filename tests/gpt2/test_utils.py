import pytest
from transformers import GPT2Tokenizer
import pandas as pd
from src.gpt2.preprocess import load_data
from src.gpt2.utils import split_data, build_dataloaders
from src.gpt2.config import MODEL_NAME


@pytest.fixture
def tokenizer():
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tok.add_special_tokens(
        {
            "pad_token": "<|pad|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
        }
    )
    return tok



class TestUtils:
    def test_split_data(self, temp_reuters_headlines):
        Description = load_data(file_path=temp_reuters_headlines)
        
        train_description, test_description = split_data(
            Description,
            test_size=0.20,
            random_state=42
        )
        
        assert not isinstance(train_description, pd.DataFrame)
        assert not isinstance(test_description, pd.DataFrame)
        
        assert len(str(train_description)) > len(str(test_description))
        
    def test_build_dataloaders(self, temp_reuters_headlines, tokenizer):
        """test loaders"""
        Description = load_data(file_path=temp_reuters_headlines)
        
        train_description, test_description = split_data(
            Description,
            test_size=0.20,
            random_state=42
        )
        
        train_loader, test_loader = build_dataloaders(
            train_description,
            test_description,
            tokenizer
        )
        
        assert "input_ids" in next(iter(train_loader))
        assert "attention_mask" in next(iter(train_loader))
        
        
        assert "input_ids" in next(iter(test_loader))
        assert "attention_mask" in next(iter(test_loader))
        
        


