import pytest
import pandas as pd
from textblob import TextBlob

from src.bert.utils import polarity, sentiment, label_encode_sentiments


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return pd.DataFrame({
        "Headlines": ["The stock market is crashing", "People are feeling confident", "The Dow Jones is open"]
    })


class TestUtils:
    """Test Utils Module for Bert"""
    def test_polarity_positive(self):
        """test polarity user-defined function"""
        
        text = "I love this movie"
        
        actual = polarity(text)
        
        assert actual > float(0.0)
        
    def test_polarity_negative(self):
        """Test polarity is less than 0.0"""
        
        text = "I hate this movie"
        
        actual = polarity(text)
        
        assert actual < float(0.0)
        
    def test_polarity_neutral(self):
        """test polarity returns neutral score"""
        
        text = "Do you want to see a movie?"
        
        actual = polarity(text)
        
        assert actual == float(0.0)


class TestSentiment:
    """Test sentiment user-defined function"""
    def test_positive_sentiment(self):
        """Test sentiment score returns positive string"""
        
        text = "I love this movie"
        
        positive_polarity = polarity(text)
        
        actual = sentiment(positive_polarity)
        
        
        expected = "Positive"
        
        
        assert actual == expected
        
    def test_neutral_sentiment(self):
        """Test that sentiment is neutral"""
        
        text = "Do you want to see a movie?"
        
        neutral_polarity = polarity(text)
        
        actual = sentiment(neutral_polarity)
        
        expected = "Neutral"
        
        assert actual == expected
        
    def test_negative_sentiment(self):
        """Test that sentiment is negative"""
        
        text = "I hate this movie!"
        
        negative_polarity = polarity(text)
        
        actual = sentiment(negative_polarity)
        
        expected = "Negative"
        
        assert actual == expected
        

class TestUtilsDf:
    """Test utils applied to dataframe"""
    
    def test_polarity(self, sample_data):
        """Test polarity applied to sample data"""
        
        sample_data['polarity'] = sample_data['Headlines'].apply(polarity)
        
        assert "polarity" in sample_data.columns
        
        assert sample_data['polarity'][0] == float(0.0)
        
        assert sample_data['polarity'][1] > float(0.0)
        
        
    def test_sentiment(self, sample_data):
        """Test sentiments applied to dataframe"""
        
        sample_data['polarity'] = sample_data['Headlines'].apply(polarity)
        
        sample_data['sentiment'] = sample_data['polarity'].apply(sentiment)
        
        
        
        assert sample_data['sentiment'][0] == "Neutral"
        assert sample_data['polarity'][0] == float(0.0)
        
        assert sample_data['sentiment'][1] == "Positive"
        
        
    def test_label_encode_sentiments(self, sample_data):
        """Test label encode function"""
        
        sample_data['polarity'] = sample_data['Headlines'].apply(polarity)
        
        sample_data['sentiment'] = sample_data['polarity'].apply(sentiment)
        
        sample_data = label_encode_sentiments(sample_data)
        
        assert pd.api.types.is_integer_dtype(sample_data['sentiment'])
        
        assert "polarity" not in sample_data.columns
        
        assert not sample_data.duplicated().any()
        
        

    
        
    
    
    
        
        
        
        
        
        

