# config.py
import os
import pandas as pd

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_dir = os.path.join(base_dir, "Data", "raw")
meta_data_dir = os.path.join(base_dir, "Data", "metadata")
processed_data_dir = os.path.join(base_dir, "Data", "processed")
results_dir = os.path.join(base_dir, "Results")
raw_data_file_path= os.path.join(raw_data_dir, "data.csv")
meta_data_file_path = os.path.join(meta_data_dir, "metadata_variabletype.csv")
columns_to_drop_file_path = os.path.join(meta_data_dir, "metadata_columns_to_drop.csv")
gender_specific_columns_file_path = os.path.join(meta_data_dir, "metadata_genderspecific_columns.csv")
process_data_file = os.path.join(processed_data_dir, "processed_data.csv")


# Load metadata and columns
metadata_columns = pd.read_csv(meta_data_file_path)
columns_to_drop = pd.read_csv(columns_to_drop_file_path)['column_name'].tolist()
numerical_columns = metadata_columns[metadata_columns['variable_type'] == 'numerical']['variable_name'].tolist()
categorical_columns = metadata_columns[metadata_columns['variable_type'] == 'categorical']['variable_name'].tolist()
target_columns = metadata_columns[metadata_columns['role'] == 'target']['variable_name'].tolist()
