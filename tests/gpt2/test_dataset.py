import pytest
from transformers import GPT2Tokenizer
from src.gpt2.preprocess import load_data
from src.gpt2.dataset import CustomDataset
from src.gpt2.config import MODEL_NAME
from src.gpt2.utils import split_data, build_dataloaders


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


class TestCustomDataset:
    def test_dataset_item_has_bos_eos(self, temp_reuters_headlines, tokenizer):
        df = load_data(file_path=temp_reuters_headlines)
        train_description, _ = split_data(df, test_size=0.20, random_state=42)

        train_set = CustomDataset(train_description, tokenizer)

        sample = train_set[0]
        assert isinstance(sample, str)
        assert sample.startswith(tokenizer.bos_token)
        assert sample.endswith(tokenizer.eos_token)

    def test_build_dataloaders_return_tokenized_batches(self, temp_reuters_headlines, tokenizer):
        df = load_data(file_path=temp_reuters_headlines)
        train_description, val_description = split_data(
            df,
            test_size=0.20,
            random_state=42,
        )

        train_loader, val_loader = build_dataloaders(
            train_description,
            val_description,
            tokenizer,
        )

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert "input_ids" in train_batch
        assert "attention_mask" in train_batch
        assert "input_ids" in val_batch
        assert "attention_mask" in val_batch
        

        
        
        
        


    
        
        
        
        







