"""Helper functions for the GPT-2 pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.gpt2.config import CONFIG
from src.gpt2.dataset import CustomDataset


def split_data(
    text: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str]]:
    """Split a description DataFrame into train/validation lists.

    Args:
        text: DataFrame containing a ``Description`` column.
        test_size: Fraction of rows reserved for validation.
        random_state: RNG seed for the split.

    Returns:
        Tuple ``(train_descriptions, val_descriptions)`` where each element is
        a Python list of strings.

    Raises:
        KeyError: If ``Description`` column is missing.
    """
    if "Description" not in text.columns:
        raise KeyError("Input DataFrame must contain a 'Description' column.")

    df_train, df_val = train_test_split(text, test_size=test_size, random_state=random_state)

    train_description = df_train["Description"].reset_index(drop=True).tolist()
    val_description = df_val["Description"].reset_index(drop=True).tolist()
    return train_description, val_description


def build_dataloaders(
    train_description: list[str],
    val_description: list[str],
    tokenizer: Any,
    batch_size: int = CONFIG.batch_size,
) -> tuple[DataLoader, DataLoader]:
    """Wrap description lists into shuffled and unshuffled DataLoaders.

    Args:
        train_description: Training descriptions.
        val_description: Validation descriptions.
        tokenizer: HuggingFace GPT-2 tokenizer used by :class:`CustomDataset`.
        batch_size: Mini-batch size; defaults to the project-wide
            ``BATCH_SIZE``.

    Returns:
        Tuple ``(train_loader, val_loader)``.
    """
    train_set = CustomDataset(Description=train_description, tokenizer=tokenizer)
    val_set = CustomDataset(Description=val_description, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_set.collate_fn,
    )
    return train_loader, val_loader
