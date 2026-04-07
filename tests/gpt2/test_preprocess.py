import pytest
import pandas as pd

from src.gpt2.preprocess import load_data



class TestPreprocess:
    """Test GPT-2 preprocess module."""

    def test_load_data_happy_path(self, temp_reuters_headlines):
        """Loads CSV, keeps Description, drops unneeded columns."""
        df = load_data(file_path=temp_reuters_headlines)

        assert isinstance(df, pd.DataFrame)
        assert "Description" in df.columns
        assert "Time" not in df.columns
        assert "Headlines" not in df.columns

    

        
        
        
        

    
    