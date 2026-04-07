MODEL_NAME: str = "bert-base-uncased"
EPOCHS: int = 4
LEARNING_RATE: float = 2e-5
DATA_PATH: str = "data/guardian_headlines.csv"
MAX_LENGTH: int = 80
BATCH_SIZE: int = 12
DEVICE: str = "cuda"
HOLDOUT_SIZE: float = 0.50
TEST_SIZE_FROM_HOLDOUT: float = 0.20
SEED: int = 42


