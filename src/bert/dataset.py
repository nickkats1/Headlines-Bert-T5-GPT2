"""PyTorch ``Dataset`` for fine-tuning BERT on headline classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Tokenizes headlines and pairs them with integer sentiment targets.

    Attributes:
        Headlines: Sequence of raw headline strings.
        targets: Integer sentiment labels aligned with ``Headlines``.
        tokenizer: HuggingFace BERT tokenizer.
        max_length: Maximum token length used for padding/truncation.
    """

    def __init__(
        self,
        Headlines: Sequence[str],
        targets: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        """Initialize the dataset.

        Args:
            Headlines: Sequence (list, ndarray, ...) of headline strings.
            targets: Integer sentiment labels of the same length as Headlines.
            tokenizer: HuggingFace BERT tokenizer.
            max_length: Maximum token length used by the tokenizer.

        Raises:
            ValueError: If ``Headlines`` and ``targets`` have different lengths.
        """
        if len(Headlines) != len(targets):
            raise ValueError(
                f"Headlines ({len(Headlines)}) and targets ({len(targets)}) "
                f"must have the same length."
            )
        self.Headlines = Headlines
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.Headlines)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Tokenize and return the example at ``index``.

        Args:
            index: Integer index of the example.

        Returns:
            A dictionary with keys ``Headlines`` (str), ``input_ids``,
            ``attention_mask`` (both 1-D tensors of length ``max_length``),
            and ``targets`` (scalar long tensor).
        """
        headline = str(self.Headlines[index])
        target = self.targets[index]

        encoded = self.tokenizer(
            headline,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "Headlines": headline,
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }
