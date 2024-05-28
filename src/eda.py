import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def summary_statistics(df):
    """
    Print summary statistics of the DataFrame.

    Parameters:
    - df (DataFrame): Pandas DataFrame to analyze.

    Returns:
    - None
    """
    try:
        print("Summary Statistics:")
        print(df.describe(include='all'))
    except Exception as e:
        print(f"Failed to generate summary statistics: {e}")

def missing_values_analysis(df):
    """
    Print the count and percentage of missing values in each column.

    Parameters:
    - df (DataFrame): Pandas DataFrame to analyze.

    Returns:
    - None
    """
    try:
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
        print("Missing Values Analysis:")
        print(missing_data[missing_data['Missing Values'] > 0])
    except Exception as e:
        print(f"Failed to analyze missing values: {e}")

def correlation_matrix(df):
    """
    Print the correlation matrix of the DataFrame.

    Parameters:
    - df (DataFrame): Pandas DataFrame to analyze.

    Returns:
    - None
    """
    try:
        corr_matrix = df.corr()
        print("Correlation Matrix:")
        print(corr_matrix)
    except Exception as e:
        print(f"Failed to generate correlation matrix: {e}")


def plot_numeric_distributions(df, columns):
    """
    Plot the distribution of numeric features.

    Parameters:
    - df (DataFrame): Pandas DataFrame to analyze.
    - columns (list of str): List of numeric column names to plot.

    Returns:
    - None
    """
    try:
        for column in columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
    except Exception as e:
        print(f"Failed to plot numeric distributions: {e}")

def plot_categorical_counts(df, columns):
    """
    Plot the count of unique values for categorical features.

    Parameters:
    - df (DataFrame): Pandas DataFrame to analyze.
    - columns (list of str): List of categorical column names to plot.

    Returns:
    - None
    """
    try:
        for column in columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=column)
            plt.title(f'Count of Unique Values in {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()
    except Exception as e:
        print(f"Failed to plot categorical counts: {e}")

def plot_pairplot(df, columns, hue=None):
    """
    Plot pairplot for a set of features.

    Parameters:
    - df (DataFrame): Pandas DataFrame to analyze.
    - columns (list of str): List of column names to include in the pairplot.
    - hue (str): Column name for color encoding. Defaults to None.

    Returns:
    - None
    """
    try:
        sns.pairplot(df[columns], hue=hue)
        plt.show()
    except Exception as e:
        print(f"Failed to plot pairplot: {e}")

def count_and_print_classes(df, target_columns):
    class_counts = {}
    for target in target_columns:
        if target in df.columns:
            counts = df[target].value_counts().to_dict()
            class_counts[target] = counts
            print(f"Class counts for {target}:")
            for class_label, count in counts.items():
                print(f"  Class {class_label}: {count}")
                print()  # Blank line for readability
        else:
                print(f"Warning: {target} not found in DataFrame columns.")
    return class_counts




# if __name__ == "__main__":
#     # Load and clean data
#     df = read_csv_data('ghana_panel_survey.csv')
#     if df is not None:
#         df = clean_data(df)
#
#         # Perform EDA
#         summary_statistics(df)
#         missing_values_analysis(df)
#         correlation_matrix(df)
#
#         numeric_columns = ['age', 'income', 'score']  # Replace with your actual numeric columns
#         plot_numeric_distributions(df, numeric_columns)
#
#         categorical_columns = ['gender', 'occupation', 'marital_status']  # Replace with your actual categorical columns
#         plot_categorical_counts(df, categorical_columns)
#
#         columns_for_pairplot = ['age', 'income', 'score', 'gender_Male']  # Replace with your actual columns
#         plot_pairplot(df, columns_for_pairplot, hue='gender_Male')

