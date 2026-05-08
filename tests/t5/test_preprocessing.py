import pandas as pd
import pytest

from src.t5.preprocess import load_data


class TestPreprocess:
    """test preprocess module"""

    def test_loads_data(self, temp_reuters_headlines):
        """test returns pd.DataFrame"""
        df = load_data(file_path=temp_reuters_headlines)

        assert isinstance(df, pd.DataFrame)
        assert df.columns is not None
        assert "Headlines" in df.columns
        assert "Description" in df.columns
        assert "Time" not in df.columns
        assert not df.columns.duplicated().any()

    def test_raises_file_not_found(self):
        """test 'FileNotFound' error raised"""
        with pytest.raises(FileNotFoundError):
            load_data(file_path="fake")
