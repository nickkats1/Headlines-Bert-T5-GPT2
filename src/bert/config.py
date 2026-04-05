import os
import pathlib

MODEL_NAME: str = "bert-base-uncased"
EPOCHS: int = 4
LEARNING_RATE: float = 2e-5
DATA_PATH: os.PathLike = pathlib.Path("data/guardian_headlines.csv")
MAX_LENGTH: int = 80
BATCH_SIZE: int = 12
DEVICE: str = "cuda"
TEST_SIZE: float = 0.20



