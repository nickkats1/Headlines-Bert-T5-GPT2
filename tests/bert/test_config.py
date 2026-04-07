import pathlib
import pytest

from src.bert import config


class TestBertConfig:
    def test_model_name(self):
        assert config.MODEL_NAME == "bert-base-uncased"

    def test_epochs(self):
        assert isinstance(config.EPOCHS, int)
        assert config.EPOCHS > 0

    def test_learning_rate(self):
        assert isinstance(config.LEARNING_RATE, float)
        assert 0 < config.LEARNING_RATE < 1

    def test_data_path_type_and_value(self):
        assert config.DATA_PATH == "data/guardian_headlines.csv"

    def test_max_length(self):
        assert isinstance(config.MAX_LENGTH, int)
        assert config.MAX_LENGTH >= 16

    def test_batch_size(self):
        assert isinstance(config.BATCH_SIZE, int)
        assert config.BATCH_SIZE > 0

    def test_device_value(self):
        assert config.DEVICE in {"cuda", "cpu", "cuda:0"}

    def test_test_size(self):
        assert config.HOLDOUT_SIZE == 0.50
        assert config.TEST_SIZE_FROM_HOLDOUT == 0.20




        
        

    
