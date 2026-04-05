from src.t5.config import model_params



class TestT5Config:
    """test config dict of T5"""
    def test_keys_exist(self):
        expected_keys = [
            "MAX_SOURCE_TEXT_LENGTH",
            "MAX_TARGET_TEXT_LENGTH",
            "TRAIN_BATCH_SIZE",
            "VALID_BATCH_SIZE",
            "TRAIN_EPOCHS",
            "VAL_EPOCHS",
            "MODEL",
            "LEARNING_RATE",
            "SEED",
        ]
        for key in expected_keys:
            assert key in model_params
            






