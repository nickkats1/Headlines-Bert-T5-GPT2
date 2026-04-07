import pytest
from transformers import T5Tokenizer


from src.t5.preprocess import load_data
from src.t5.config import SOURCE_LENGTH, TARGET_LENGTH, MODEL_NAME
from src.t5.dataset import CustomDataset


@pytest.fixture
def tokenizer():
    return T5Tokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def custom_dataset(temp_reuters_headlines, tokenizer):
    df = load_data(file_path=temp_reuters_headlines)
    return CustomDataset(
        dataframe=df,
        tokenizer=tokenizer,
        source_len=SOURCE_LENGTH,
        target_len=TARGET_LENGTH,
        source_col="Description",
        target_col="Headlines",
    )



class TestCustomDataset:
    """test custom dataset"""
    def test_attributes(self, custom_dataset):
        """test attributes from custom dataset."""
        assert isinstance(custom_dataset, CustomDataset)
        assert custom_dataset.source_len == 128
        assert custom_dataset.target_len == 32
        assert hasattr(custom_dataset, "tokenizer")
        assert hasattr(custom_dataset, "data")
        assert hasattr(custom_dataset, "source_text")
        assert hasattr(custom_dataset, "target_text")
        
    def test_len(self, custom_dataset):
        assert len(custom_dataset) > 0
        
    def test_getitem_key(self, custom_dataset):
        sample = custom_dataset[0]
        
        assert sample["source_ids"].shape == (SOURCE_LENGTH,)
        assert sample["source_mask"].shape == (SOURCE_LENGTH,)
        assert sample['target_ids'].shape == (TARGET_LENGTH,)
        assert sample['target_ids'].shape == (TARGET_LENGTH,)
        
        assert "source_ids" in sample
        assert "target_ids" in sample
        assert "target_ids" in sample
        
        
        