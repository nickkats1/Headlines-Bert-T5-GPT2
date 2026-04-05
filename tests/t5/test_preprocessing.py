import pandas as pd
from src.t5.preprocess import load_data



class TestLoadData:
    """Test Load data"""
    def test_returns_dataframe(self, temp_reuters_headlines):
        """test if 'load_data' returns pd.DataFrame"""
        
        df = load_data(temp_reuters_headlines)
        
        assert isinstance(df, pd.DataFrame)
        assert df.columns is not None
        
    def test_columns_exist(self, temp_reuters_headlines):
        """Test if proper columns are returns"""
        
        df = load_data(temp_reuters_headlines)
        
        assert "Headlines" in df.columns
        assert "Description" in df.columns
        assert "Time" not in df.columns
        assert len(df.columns) == int(2)
        assert not df.duplicated().any()
        

        
        




    




