from src.gpt2 import config



class TestConfig:
    """Test GPT2 Config"""
    def test_model_name(self):
        """Test model name is in config"""
        assert config.MODEL_NAME == "gpt2"
    
    def test_epochs(self):
        """test if epochs are in config"""
        assert config.EPOCHS == 3
    
    def test_device(self):
        """Test that device is in config"""
        assert config.DEVICE == "cuda"
        
    def test_learning_rate(self):
        """test learning rate is in config"""
        assert config.LEARNING_RATE == 5e-5
 
    



