import pandas as pd
import os
from sklearn.model_selection import train_test_split
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


# Function to split data and save to CSV files
def split_train_test(df, test_size=0.2, random_state=42, output_folder='processed'):
    """
    Splits the DataFrame into a single training and testing set, saves them to CSV files, and returns them.

    Parameters:
    - df (DataFrame): The DataFrame to split.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.
    - output_folder (str): The folder to save the CSV files.

    Returns:
    - train_df (DataFrame): The training set DataFrame.
    - test_df (DataFrame): The testing set DataFrame.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split data once
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    train_filename = os.path.join(output_folder, 'train_data.csv')
    test_filename = os.path.join(output_folder, 'test_data.csv')
    print(train_filename)
    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)

    print(f"Saved train data to {train_filename}")
    print(f"Saved test data to {test_filename}")

    return train_df, test_df


def save_to_csv(df, directory, filename,index=False):
    """
    Saves a DataFrame to a CSV file in the specified directory.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to save.
    - directory (str): The directory where the CSV file will be saved.
    - filename (str): The name of the CSV file.

    Returns:
    - None
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")

    # Full path to save the file
    file_path = os.path.join(directory, filename)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=index)
    print(f"Data saved to {file_path}")
