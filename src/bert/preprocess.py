import pandas as pd



# --- load data from file path

def load_data(file_path: str) -> pd.DataFrame:
    """returns CSV file as pd.DataFrame.
    
    Args:
        file_path: path of CSV file.
    
    Returns:
        pd.Dataframe: A dataframe containing CSV contents.
        
    Raises:
        - FileNotFoundError:
            - Raised if file does not exits in file path
    """
    if file_path is None:
        raise FileNotFoundError("Could Not find file in path")
    return pd.read_csv(file_path, delimiter=",")




# --- clean dataframe ---

def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe and return Headlines column only.
    
    Args:
        dataframe: Dataframe containing 'Headlines' column.
        
    Returns:
        dataframe: A dataframe with no duplicated values containing only 'Headlines' column
    """
    dataframe.drop('Time', axis=1, inplace=True)
    dataframe.drop_duplicates(inplace=True)
    return dataframe