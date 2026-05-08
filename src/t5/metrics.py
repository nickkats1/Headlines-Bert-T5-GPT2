"""ROUGE evaluation and prediction-saving helpers for the T5 pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


def compute_rouge(predictions: Sequence[str], actuals: Sequence[str]) -> dict[str, float]:
    """Compute average ROUGE-1/2/L F1 over prediction-reference pairs.

    Args:
        predictions: Generated headline strings.
        actuals: Ground-truth headline strings.

    Returns:
        Mapping ``{"rouge1": ..., "rouge2": ..., "rougeL": ...}`` of average
        F1 scores, all zero when ``predictions`` is empty.

    Raises:
        ValueError: If ``predictions`` and ``actuals`` differ in length.
    """
    if len(predictions) != len(actuals):
        raise ValueError(
            f"predictions ({len(predictions)}) and actuals ({len(actuals)}) "
            f"must have the same length."
        )

    n = len(predictions)
    if n == 0:
        return dict.fromkeys(ROUGE_KEYS, 0.0)

    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)
    totals = dict.fromkeys(ROUGE_KEYS, 0.0)

    for pred, actual in zip(predictions, actuals, strict=True):
        scores = scorer.score(str(actual), str(pred))
        for key in ROUGE_KEYS:
            totals[key] += scores[key].fmeasure

    results = {key: totals[key] / n for key in ROUGE_KEYS}
    print("  " + " | ".join(f"{key.upper()}: {results[key]:.4f}" for key in ROUGE_KEYS))
    return results


def save_predictions(
    predictions: Sequence[str],
    actuals: Sequence[str],
    path: str | Path = "predictions.csv",
) -> Path:
    """Save generated and reference texts side-by-side to a CSV file.

    Args:
        predictions: Generated text strings.
        actuals: Ground-truth text strings.
        path: Destination CSV path.

    Returns:
        The resolved ``Path`` written to.

    Raises:
        ValueError: If ``predictions`` and ``actuals`` differ in length.
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have the same length.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Generated Text": list(predictions), "Actual Text": list(actuals)}).to_csv(
        out_path, index=False
    )
    print(f"  Predictions saved to {out_path}")
    return out_path
