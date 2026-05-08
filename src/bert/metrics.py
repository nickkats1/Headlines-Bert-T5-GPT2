"""Classification metrics and inference helpers for the BERT pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Accuracy in ``[0.0, 1.0]``.
    """
    return float(accuracy_score(y_true, y_pred))


def compute_f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute weighted F1 score.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Weighted F1 score in ``[0.0, 1.0]``.
    """
    return float(f1_score(y_true, y_pred, average="weighted"))


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        2-D ndarray confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)


def get_predictions(
    model: torch.nn.Module, dataloader, device
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Run ``model`` over ``dataloader`` and collect predictions and labels.

    Args:
        model: Trained BERT classifier.
        dataloader: DataLoader yielding the dict produced by ``CustomDataset``.
        device: Torch device used for inference.

    Returns:
        Tuple of (headlines, y_pred, y_true) where ``headlines`` is a list of
        raw strings and the prediction/label tensors are 1-D ``long`` tensors.
    """
    model.eval()

    headlines: list[str] = []
    y_pred: list[int] = []
    y_true: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            headlines.extend(batch["Headlines"])
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(targets.cpu().tolist())

    return headlines, torch.tensor(y_pred, dtype=torch.long), torch.tensor(y_true, dtype=torch.long)
