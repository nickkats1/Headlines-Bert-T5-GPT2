"""Utility helpers for the BERT sentiment-classification pipeline."""

from __future__ import annotations

import pandas as pd
from textblob import TextBlob

SENTIMENT_LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def polarity(text: str) -> float:
    """Compute the TextBlob polarity score for a string.

    Args:
        text: Free-form text. Falsy or non-string input returns ``0.0``.

    Returns:
        Polarity score in ``[-1.0, 1.0]``.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    return float(TextBlob(text).polarity)


def sentiment(score: float) -> str:
    """Map a polarity score to a discrete sentiment label.

    Args:
        score: Polarity score in ``[-1.0, 1.0]``.

    Returns:
        One of ``"Negative"``, ``"Neutral"``, ``"Positive"``.
    """
    if score == 0:
        return "Neutral"
    if score < 0:
        return "Negative"
    return "Positive"


def label_encode_sentiments(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Encode the ``sentiment`` column as integers and drop helpers.

    Maps ``Negative`` → 0, ``Neutral`` → 1, ``Positive`` → 2 and drops the
    ``polarity`` helper column if present. Removes duplicate rows.

    Args:
        dataframe: DataFrame containing a ``sentiment`` column with string
            labels.

    Returns:
        A copy of ``dataframe`` with integer-encoded sentiment, the
        ``polarity`` column removed, duplicates dropped, and the index reset.

    Raises:
        KeyError: If ``sentiment`` column is missing.
    """
    if "sentiment" not in dataframe.columns:
        raise KeyError("DataFrame must contain a 'sentiment' column.")

    df = dataframe.copy()
    df["sentiment"] = df["sentiment"].map(SENTIMENT_LABEL_MAP).astype("int64")

    if "polarity" in df.columns:
        df = df.drop(columns=["polarity"])

    return df.drop_duplicates().reset_index(drop=True)
