import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split


def preprocess_data(df, categorical_columns):
    """
    Encode categorical variables using one-hot encoding.

    Parameters:
    - df (DataFrame): Pandas DataFrame to be preprocessed.
    - categorical_columns (list of str): List of column names to be encoded.

    Returns:
    - DataFrame: Preprocessed pandas DataFrame.
    """
    try:
        df_preprocessed = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        print("Data preprocessed successfully.")
        return df_preprocessed
    except Exception as e:
        print(f"Failed to preprocess data: {e}")
        return None


def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the DataFrame into train and test sets.

    Parameters:
    - df (DataFrame): Pandas DataFrame to be split.
    - target_column (str): Name of the target column.
    - test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.
    - random_state (int): Controls the shuffling applied to the data before applying the split. Defaults to 42.

    Returns:
    - tuple: Tuple containing train and test sets (X_train, X_test, y_train, y_test).
    """
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Data split successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Failed to split data: {e}")
        return None, None, None, None

def summarize_dataframe_columns(df):
    """
    Summarizes the characteristics of each column in a pandas DataFrame, including data types,
    the number of null values, and the count of unique values. The summary is presented both as
    a printed table and returned as a new DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
    - pd.DataFrame: A DataFrame containing the summary analysis of each column, including:
        - 'Column Name': Names of the columns.
        - 'Data Type': Data type of the column.
        - 'Data Type Name': Name of the data type.
        - 'No of Null Values': Number of null values in the column.
        - 'Percentage of Null Values': Percentage of total entries that are null for the column.
        - 'No of Unique Values': Number of unique values in the column.

    If the input DataFrame is empty, it prints a message and returns an empty DataFrame.
    """
    if len(df) == 0:
        print("The DataFrame is empty.")
        return pd.DataFrame()  # Return an empty DataFrame or appropriate response

    column_analysis = [
        [
            column,
            df[column].dtype,
            df[column].dtype.name,
            df[column].isnull().sum(),
            (df[column].isnull().sum() / len(df) * 100),
            df[column].nunique()
        ]
        for column in df.columns
    ]

    # Sorting the list by the percentage of null values in descending order
    column_analysis_sorted = sorted(column_analysis, key=lambda x: x[4], reverse=True)

    # Tabulating the results
    column_headers = ["Column Name", "Data Type", "Data Type Name", "No of Null Values", "Percentage of Null Values",
                      "No of Unique Values"]
    column_analysis_table = tabulate(column_analysis_sorted, headers=column_headers, tablefmt="grid")

    print(column_analysis_table)
    return pd.DataFrame(column_analysis_sorted, columns=column_headers)
