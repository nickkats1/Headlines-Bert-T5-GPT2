import pytest
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd


from src.gpt2.dataset import CustomDataset



@pytest.fixture
def tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

@pytest.fixture
def gpt2_model():
    return GPT2LMHeadModel.from_pretrained("gpt2")





class TestCustomDataset:
    """test custom dataset"""
    def test_description(self, temp_reuters_headlines, tokenizer):
        """Test Description is in CustomDataset"""
        
        df = pd.read_csv(temp_reuters_headlines, delimiter=",")
        
        df.drop(['Headlines', 'Time'], inplace=True, axis=1)
        
        df.drop_duplicates(inplace=True)
        
        from sklearn.model_selection import train_test_split
        
        Description = df['Description'].to_list()
        
        df_train, df_test = train_test_split(Description, test_size=0.20, random_state=42)
        
        
        
        train_set = CustomDataset(
            Description=df_train,
            tokenizer=tokenizer
        )
        
        test_set = CustomDataset(
            Description=df_test,
            tokenizer=tokenizer
        )
        

        
        
        train_loader = DataLoader(
            train_set,
            batch_size=2,
            collate_fn=train_set.collate_fn
        )
        
        test_loader = DataLoader(
            test_set,
            batch_size=2,
            collate_fn=test_set.collate_fn
        )
        
        assert "input_ids" in next(iter(train_loader))
        
        assert "attention_mask" in next(iter(train_loader))
        
        
        assert "input_ids" in next(iter(test_loader))
        assert "attention_mask" in next(iter(test_loader))
        
        
        sample = train_set[0]
        assert sample.startswith(tokenizer.bos_token)
        assert sample.endswith(tokenizer.eos_token)
        
        assert tokenizer.bos_token == "<|startoftext|>"
        
        assert tokenizer.pad_token == "<|pad|>"
        assert tokenizer.bos_token == "<|startoftext|>"
        
        
        
        


    
        
        
        
        







