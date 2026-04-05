import pandas as pd
from sklearn.model_selection import train_test_split


# --- split data ---

def split_data(
    text: list[str],
    test_size:float,
    random_state: int,
):
    """Split dataframe into training test splits"""
    
    df_train, df_test = train_test_split(
        text,
        test_size=test_size,
        random_state=random_state
    )

    train_description = df_train['Description'].reset_index(drop=True).to_list()
    test_description = df_test['Description'].reset_index(drop=True).to_list()
    
    return train_description, test_description



# --- convert to dataloaders


def convet_to_dataloader()



