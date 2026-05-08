from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path | None) -> pd.DataFrame:
    """Load Reuters CSV and return only the 'Description' column.

    Drops the 'Time' and 'Headlines' columns (if present), removes rows with
    missing descriptions, and de-duplicates.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with a single 'Description' column.

    Raises:
        FileNotFoundError: If file_path is None or does not exist.
    """
    if file_path is None:
        raise FileNotFoundError("file_path must not be None.")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    dataframe = pd.read_csv(path, delimiter=",")

    columns_to_drop = [c for c in ("Time", "Headlines") if c in dataframe.columns]
    if columns_to_drop:
        dataframe = dataframe.drop(columns=columns_to_drop)

    if "Description" in dataframe.columns:
        dataframe = dataframe.dropna(subset=["Description"])

    dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    return dataframe
