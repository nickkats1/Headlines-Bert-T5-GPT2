import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV file and return pd.DataFrame.
    
    Args:
        file_path: path where CSV file is located.
    
    Returns:
        pd.DataFrame: A dataframe containing both Headlines and Description columns
    
    Raises:
        FileNotFoundError:
        - Raised if file path does not exits
    """
    if file_path is not None:
        df = pd.read_csv(file_path, delimiter=",")
        df.drop("Time", axis=1, inplace=True)
        df.fillna("", inplace=True)
        df.drop_duplicates(inplace=True)
        return df
    else:
        raise FileNotFoundError("Could not find file path")