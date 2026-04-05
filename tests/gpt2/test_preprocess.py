import pytest
import pandas as pd

from src.gpt2.preprocess import load_data



class TestLoadData:
    def test_load(self, temp_reuters_headlines):
        """Test that data is returned as correct typed"""
        
        df = load_data(file_path=temp_reuters_headlines)
        
        assert df is not None
        
        assert isinstance(df, pd.DataFrame)
        
        assert "Description" in df.columns
        
        assert "Time" not in df.columns
        
        assert "Headlines" not in df.columns
        

        
    def test_raises_error(self):
        """Test raises 'FileNotFoundError' when no file is located"""
        with pytest.raises(FileNotFoundError):
            load_data(file_path=None)
            
    

        
        
        
        

    
    