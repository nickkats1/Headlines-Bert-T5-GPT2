"""Configuration for the GPT-2 fine-tuning pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    """Hyperparameters for the GPT-2 fine-tuning pipeline.

    Attributes:
        model_name: HuggingFace model identifier.
        data_path: Path (relative to repo root) to the input CSV.
        output_dir: Directory for saved model files and metrics.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        max_length: Token length for padding/truncation.
        device: Preferred device string (resolved at runtime).
        seed: RNG seed for reproducibility.
        test_size: Fraction of the dataset used for validation.
        pad_token: Pad token added to the GPT-2 tokenizer.
        bos_token: Beginning-of-sequence token added to the tokenizer.
        eos_token: End-of-sequence token added to the tokenizer.
    """

    model_name: str = "gpt2"
    data_path: str = "data/reuters_headlines.csv"
    output_dir: str = "src/gpt2/artifacts/"
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 12
    max_length: int = 128
    device: str = "cuda"
    seed: int = 42
    test_size: float = 0.20
    pad_token: str = "<|pad|>"
    bos_token: str = "<|startoftext|>"
    eos_token: str = "<|endoftext|>"


CONFIG = GPT2Config()
