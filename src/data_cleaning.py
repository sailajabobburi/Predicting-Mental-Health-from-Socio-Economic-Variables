def clean_data(df):
    """
    Perform basic data cleaning on the DataFrame.

    Parameters:
    - df (DataFrame): Pandas DataFrame to be cleaned.

    Returns:
    - DataFrame: Cleaned pandas DataFrame.
    """
    try:
        # Drop rows with missing values
        df_cleaned = df.dropna()
        # Remove duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()
        print("Data cleaned successfully.")
        return df_cleaned
    except Exception as e:
        print(f"Failed to clean data: {e}")
        return None