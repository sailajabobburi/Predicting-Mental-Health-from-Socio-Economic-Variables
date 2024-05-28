import pandas as pd
from tabulate import tabulate
from src.data_loading import read_csv_data
from sklearn.impute import SimpleImputer
import numpy as np
from src.eda import count_and_print_classes
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Extract numerical and categorical columns


def drop_columns(df, columns_to_drop):
    """
    Drops specified columns from the DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame from which to drop columns.
    - columns_to_drop (list of str): List of column names to drop.

    Returns:
    - DataFrame: The DataFrame with specified columns dropped.
    """
    df_dropped = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Dropped columns: {columns_to_drop}")
    return df_dropped


def handle_gender_specific_columns(df, gender_column, gender_specific_columns_file):
    """
    Handles gender-specific columns by filtering non-applicable records and filling with a specified value,
    and adds a binary applicability column.

    Parameters:
    - df (DataFrame): The DataFrame to process.
    - gender_column (str): The name of the gender column.
    - gender_specific_columns (list of str): List

    Returns:
    - DataFrame: The DataFrame with gender-specific columns handled.
    """
    # Read gender-specific columns from CSV file
    gender_specific_metadata = pd.read_csv(gender_specific_columns_file)

    # Filter out rows in metadata where 'gender_specific' is NaN
    gender_specific_columns = gender_specific_metadata.dropna(subset=['gender_specific'])

    # Iterate over the gender-specific columns
    for _, row in gender_specific_columns.iterrows():
        col = row['variable_name']
        gender = row['gender_specific']

        # Add a binary column for applicability
        applicability_column = f"{col}_is_applicable"
        df[applicability_column] = df[gender_column].apply(
            lambda x: 1 if (x == 1 and gender == 'female') or (x == 0 and gender == 'male') else 0)

        # Fill non-applicable records with a specified value (e.g., 3 for NA category)
        if col == 'total_fertility1':
            # For total_fertility, use NaN for males to indicate non-applicability
            df.loc[df[gender_column] == 0, col] = -1 # 0 represents male
        else:
            # Fill non-applicable records with a specified value (e.g., 3)
            if gender == 'female':
                df.loc[df[gender_column] == 0, col] = 3  # 0 for male
            elif gender == 'male':
                df.loc[df[gender_column] == 1, col] = 3  # 1 for female

    return df

def analyze_column(df, column):
    """
    Analyze a single column of the DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the column.
    - column (str): The name of the column to analyze.

    Returns:
    - list: Analysis results for the column.
    """
    no_of_rows = len(df)
    data_type = df[column].dtype
    no_of_null_values = df[column].isnull().sum()
    percentage_of_null_values = (no_of_null_values / no_of_rows) * 100
    no_of_unique_values = df[column].nunique()
    return [column, data_type, no_of_null_values, percentage_of_null_values, no_of_unique_values]

def analyze_dataframe(df):
    """
    Analyze the DataFrame and print the results in a tabulated format.

    Parameters:
    - df (DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """
    column_headers = ["Column Name", "Data Type", "No of Null Values", "Percentage of Null Values", "No of Unique Values"]
    column_analysis = []

    for column in df.columns:
        column_analysis.append(analyze_column(df, column))
        # Convert to DataFrame for easy sorting
    column_analysis_df = pd.DataFrame(column_analysis, columns=column_headers)

    # Sort by percentage of null values in descending order
    column_analysis_df = column_analysis_df.sort_values(by="Percentage of Null Values", ascending=False)

    # Convert back to list of lists for tabulate
    column_analysis_sorted = column_analysis_df.values.tolist()

    column_analysis_table = tabulate(column_analysis_sorted, column_headers, tablefmt="grid")
    print(column_analysis_table)


def handle_missing_data(df, numerical_columns,categorical_columns):
    """
    Preprocess the data based on metadata.

    Parameters:
    - df (DataFrame): The DataFrame to preprocess.
    - metadata (DataFrame): Metadata containing information about the variables.

    Returns:
    - DataFrame: Preprocessed DataFrame.
    """

    # Convert numerical columns to appropriate data types
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values for numerical columns with mean
    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

    # Round numerical columns to 0 decimal places
    df[numerical_columns] = np.round(df[numerical_columns], 0).astype(int)

    # Impute missing values for categorical columns with median
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    # Convert categorical columns back to integer type
    for col in categorical_columns:
        df[col] = df[col].astype(int)

    return df
def normalize_features(df, numerical_columns):
    """
    Apply Min-Max scaling to numerical features in the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - numerical_columns (list): List of numerical column names to be normalized.

    Returns:
    - DataFrame: DataFrame with normalized numerical features.
    """
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def apply_smote(X_train, y_train, sampling_strategy='auto', random_state=42):
    """
    Applies SMOTE to oversample the minority class in the training data.

    Parameters:
    - X_train (DataFrame): Features of the training set.
    - y_train (Series): Target variable of the training set.
    - sampling_strategy (str or float): The sampling strategy to use. Default is 'auto'.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_res (DataFrame): The resampled features of the training set.
    - y_res (Series): The resampled target variable of the training set.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res









