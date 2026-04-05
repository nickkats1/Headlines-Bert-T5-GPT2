import pathlib

from src.bert import config




class TestBertConfig:
    """Test Config for Bert"""
    
    def test_model_name(self):
        """Test model name in config"""
        assert config.MODEL_NAME == 'bert-base-uncased'

    def test_epochs(self):
        """Test epochs in config"""
    
        assert isinstance(config.EPOCHS, int)
        assert config.EPOCHS == 4
        
    def test_learning_rate(self):
        """Test learning rate in config"""
        
        assert isinstance(config.LEARNING_RATE, float)
        assert config.LEARNING_RATE == 2e-5
        
    def test_data_path(self):
        """Test data_path in config"""
        assert isinstance(config.DATA_PATH, pathlib.Path)
        
    def test_max_length(self):
        """Test max length in config"""
        assert isinstance(config.MAX_LENGTH, int)
        assert config.MAX_LENGTH == int(80)
        
        
    def test_batch_size(self):
        """Test batch size in config"""
        assert isinstance(config.BATCH_SIZE, int)
        assert config.BATCH_SIZE == 12
    
    
    def test_device(self):
        """Test device in config"""
        assert config.DEVICE == "cuda"
        
    def test_test_size(self):
        """test 'test_size' is in config"""
        assert isinstance(config.TEST_SIZE, float)
        assert config.TEST_SIZE == float(0.20)





        
        

    
