"""PyTorch ``Dataset`` for fine-tuning GPT-2 on Reuters descriptions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Wraps description strings with BOS/EOS tokens for GPT-2 fine-tuning.

    Attributes:
        Description: Sequence of raw description strings.
        tokenizer: HuggingFace GPT-2 tokenizer (must define ``bos_token`` and
            ``eos_token``).
        max_length: Maximum token length used in :meth:`collate_fn`.
    """

    def __init__(
        self,
        Description: Sequence[str],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        """Initialize the dataset.

        Args:
            Description: Sequence of description strings.
            tokenizer: HuggingFace GPT-2 tokenizer.
            max_length: Maximum token length used during collation.
        """
        self.Description = Description
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.Description)

    def __getitem__(self, index: int) -> str:
        """Return the BOS/EOS-wrapped string at ``index``."""
        text = str(self.Description[index])
        return f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"

    def collate_fn(self, batch: list[str]) -> dict[str, Any]:
        """Tokenize and pad a batch of strings.

        Args:
            batch: List of BOS/EOS-wrapped strings produced by ``__getitem__``.

        Returns:
            Dictionary with ``input_ids`` and ``attention_mask`` tensors.
        """
        return self.tokenizer(
            batch,
            padding="longest",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
