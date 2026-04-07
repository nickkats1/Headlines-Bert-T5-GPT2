from src.t5 import config


class TestConfig:
    """Test config module"""
    def test_source_length(self):
        """test that source length is in config"""
        assert config.SOURCE_LENGTH is not None
        assert config.SOURCE_LENGTH == 128

    def test_target_len(self):
        """test target length in config"""
        assert config.TARGET_LENGTH is not None
        assert config.TARGET_LENGTH == 32
        
    def test_batch_size(self):
        """test batch size in config"""
        assert config.BATCH_SIZE == 12

    def test_epochs(self):
        """test epochs in config"""
        assert config.EPOCHS == 2
        
    def test_model_name(self):
        """test model name in config"""
        assert config.MODEL_NAME == "t5-base"
        
    def test_learning_rate(self):
        """test learning rate in config"""
        assert config.LEARNING_RATE == 5e-5
        
    def test_file_path(self):
        """test file path in config"""
        assert config.DATA_PATH == "data/reuters_headlines.csv"
        
    def test_device(self):
        """test device in config"""
        assert config.DEVICE == "cuda:0"
        
    def test_output_dir(self):
        assert config.OUTPUT_DIR == "src/t5/artifacts/"
        
    
        
        