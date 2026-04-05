import os
from pathlib import Path

MODEL_NAME: str = "gpt2"
EPOCHS: int = 3
LEARNING_RATE: float = 5e-5
DATA_PATH: os.PathLike = Path("data/reuters_headlines.csv")
MAX_LENGTH: int = 128
BATCH_SIZE: int = 12
DEVICE: str = "cuda"