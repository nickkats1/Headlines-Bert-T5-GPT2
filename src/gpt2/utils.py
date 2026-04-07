import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


from src.gpt2.dataset import CustomDataset
from src.gpt2.config import BATCH_SIZE


# --- split data ---

def split_data(
    text: pd.DataFrame,
    test_size:float,
    random_state: int,
):
    """Split dataframe into training test splits"""
    
    df_train, df_val = train_test_split(
        text,
        test_size=test_size,
        random_state=random_state
    )

    train_description = df_train['Description'].reset_index(drop=True).to_list()
    val_description = df_val['Description'].reset_index(drop=True).to_list()
    
    return train_description, val_description



# --- convert to dataloaders


def build_dataloaders(train_description, val_description, tokenizer):
    """Created train and test dataloaders from 'CustomDataset'
    
    Args:
        train_description: 80% of the 'Description' text.
        test_description: 20% of the 'Description' text.
        
    Returns:
        train_loader: train_description with dataloader wrapper
        test_loader: test description with dataloader wrapper.
    """
    train_set = CustomDataset(
        Description=train_description,
        tokenizer=tokenizer,
    )

    val_set = CustomDataset(
        Description=val_description,
        tokenizer=tokenizer,
    )
    
    # get loaders
    
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_set.collate_fn
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_set.collate_fn
    )
    
    return train_loader, val_loader

    


