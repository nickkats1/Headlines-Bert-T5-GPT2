import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A DataFrame containing CSV contents.

    Raises:
        FileNotFoundError: If file_path is None.
    """
    if file_path is None:
        raise FileNotFoundError("Could not find file path.")

    return pd.read_csv(file_path, delimiter=",")


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean raw BERT dataframe.

    Steps:
        1. Remove optional 'Time' column if present.
        2. Drop duplicate rows.

    Args:
        dataframe: Input DataFrame.

    Returns:
        A cleaned copy of the DataFrame.
    """
    cleaned = dataframe.copy()

    if "Time" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Time"])

    cleaned = cleaned.drop_duplicates()

    return cleaned