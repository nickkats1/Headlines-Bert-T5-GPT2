import pandas as pd



def load_data(file_path: str) -> pd.DataFrame:
    """Load data from file path and return string
    consisting of 'Descriptions' text data.
    
    Args:
        file_path: path of CSV file for ingestion.
        
    Returns:
        Description: pd.DataFrame consisting of descriptions from 
        rutgers headlines
    """
    if file_path is not None:
        dataframe = pd.read_csv(file_path, delimiter=",")
        dataframe.drop(['Time', 'Headlines'], inplace=True, axis=1)
        dataframe.drop_duplicates(inplace=True)
        return dataframe
    else:
        raise FileNotFoundError("Could Not find file path")