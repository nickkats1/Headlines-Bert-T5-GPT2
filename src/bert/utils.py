import pandas as pd
from textblob import TextBlob
from torch.utils.data import DataLoader

def polarity(text: str) -> float:
    """polarity score applied to selected test column
    
    Args:
        text: corpus of text data.
        
    Returns:
        polarity score: Score ranging from -1 to 1 to score polarity
    """
    return TextBlob(text).polarity


def sentiment(text: float) -> str:
    """returns polarity scores as discrete categorical values.
    
    Args:
        text: text with polarity score applied.
    
    Returns:
        sentiment: polarity score converted to sentiment values
    """
    
    if text == 0:
        return "Neutral"
    elif text < 0:
        return "Negative"
    else:
        return "Positive"


def label_encode_sentiments(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Label encode sentiments as discrete values for training.

    Args:
        dataframe: DataFrame with 'sentiment' column containing string labels.

    Returns:
        pd.DataFrame: DataFrame with sentiment mapped to integers.
    """
    dataframe = dataframe.copy()
    dataframe['sentiment'] = dataframe['sentiment'].map({
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    })
    
    dataframe.drop("polarity", axis=1, inplace=True)
    dataframe.drop_duplicates(inplace=True)
    
    return dataframe

