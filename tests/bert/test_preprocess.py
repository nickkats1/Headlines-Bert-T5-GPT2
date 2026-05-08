import pandas as pd
import pytest

from src.bert.preprocess import clean_data, load_data


class TestPreprocess:
    """Test Load Data"""

    def test_raises_error(self):
        """test FileNotFoundError is raised"""
        with pytest.raises(FileNotFoundError):
            load_data(file_path=None)

    def test_load_data(self, temp_guardian_file):
        """test the data is loaded"""
        df = load_data(file_path=temp_guardian_file)

        assert "Headlines" in df.columns
        assert isinstance(df, pd.DataFrame)

    def test_clean_data(self, temp_guardian_file):
        """Test that clean data returns pd.DataFrame with no duplicates"""

        df = load_data(file_path=temp_guardian_file)

        df_cleaned = clean_data(df)

        assert isinstance(df_cleaned, pd.DataFrame)

        assert "Headlines" in df_cleaned.columns

    def test_clean_data_drops_column(self, temp_guardian_file):
        """Test Clean data drops 'Time' column"""

        df = load_data(file_path=temp_guardian_file)

        df_cleaned = clean_data(df)

        assert "Time" not in df_cleaned.columns

        assert "Headlines" in df.columns

    def test_duplicates(self, temp_guardian_file):
        """Test if any duplicates in clean data function"""

        df = load_data(file_path=temp_guardian_file)

        df_cleaned = clean_data(df)

        assert not df_cleaned.duplicated().any()
