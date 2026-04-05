import pytest
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd



def split_data(
    text: list[str],
    test_size:float,
    random_state: int,
):
    """Split dataframe into training test splits"""
    
    df_train, df_test = train_test_split(
        text,
        test_size=test_size,
        random_state=random_state
    )

    train_description = df_train['Description'].reset_index(drop=True).to_list()
    test_description = df_test['Description'].reset_index(drop=True).to_list()
    
    return train_description, test_description
    
    


class TestUtils:
    """Test util functions"""
    def test_split_data(self, temp_reuters_headlines):
        """Test split data user-defined function"""
        df = pd.read_csv(temp_reuters_headlines, delimiter=",")
        

        
        assert df.columns is not None
        
        
        train_description, test_description = split_data(df, test_size=0.20, random_state=42)
        
        assert len(train_description) > len(test_description)
        
        assert "Headlines" not in train_description
        assert "Headlines" not in test_description
        
        
        
        
        
        


