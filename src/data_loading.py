import pandas as pd
def read_csv_data(filepath, delimiter=',', encoding='utf-8', index_col=None, parse_dates=False):
    """
    Read data from a CSV file into a pandas DataFrame.

    Parameters:
    - filepath (str): Path to the CSV file.
    - delimiter (str): Delimiter used in the CSV file. Defaults to ','.
    - encoding (str): Encoding of the CSV file. Defaults to 'utf-8'.
    - index_col (int, str, sequence of int / str, or False): Column(s) to set as index(MultiIndex). Defaults to None.
    - parse_dates (bool or list of int / names): Attempt to parse data to datetime; either specify True or a list of the columns to parse. Defaults to False.

    Returns:
    - DataFrame: Pandas DataFrame containing the data from the CSV file.
    """
    try:
        # Load the dataset
        data = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding,
                           index_col=index_col, parse_dates=parse_dates)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None
