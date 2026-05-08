from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path | None) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A DataFrame containing CSV contents.

    Raises:
        FileNotFoundError: If file_path is None or does not exist.
    """
    if file_path is None:
        raise FileNotFoundError("file_path must not be None.")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path, delimiter=",")


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean raw BERT dataframe.

    Steps:
        1. Remove optional 'Time' column if present.
        2. Drop rows missing the 'Headlines' column value.
        3. Drop duplicate rows.

    Args:
        dataframe: Input DataFrame.

    Returns:
        A cleaned copy of the DataFrame with a fresh 0..N-1 index.
    """
    cleaned = dataframe.copy()

    if "Time" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Time"])

    if "Headlines" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["Headlines"])

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned
