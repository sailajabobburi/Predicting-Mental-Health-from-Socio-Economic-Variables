import pandas as pd
from src.data_preprocessing import drop_columns
# Define the path to the CSV file
csv_file_path = '../Data/raw/data.csv'

# Read the CSV file to get the column headers
df = pd.read_csv(csv_file_path, nrows=0)  # Read only the header



# List of columns to mark as numerical
numerical_columns = [
    "age_married1", "prop_U181", "prop_adult1", "prop_E60_691", "prop_70plus1",
    "prop_ill_psttwo1", "bmi1", "obese1", "height1", "prop_hh_nhis1",
    "big5open_score1", "big5consc_score1", "big5extrav_score1", "big5agreeable_score1", "big5neurotic_score1",
    "sleeping_time1", "k61", "k62", "k63", "k64", "kessler1", "kessler2", "kessler3", "kessler4",
    "age_yrs1", "age_sq_1001"
]

target_columns = ["kessler_dummy2",	"kessler_dummy3","kessler_dummy4"]

dropped_columns=["k61","k62","k63","k64","kessler1","kessler2","kessler3","kessler4","exit_entry_w1w2","exit_entry_w1w3","exit_entry_w1w4","exit_entry_w2w3","exit_entry_w3w4"]

df= drop_columns(df, dropped_columns)
# Create a metadata DataFrame
metadata = pd.DataFrame({
    "variable_name": df.columns,
    "variable_type": ["numerical" if col in numerical_columns else "categorical" for col in df.columns],
    "role": ["target" if col in target_columns else "feature" for col in df.columns]

})

# Save the metadata to a CSV file
metadata_file_path = '../Data/processed/metadata_variabletype_generated.csv'
metadata.to_csv(metadata_file_path, index=False)

metadata.head(), metadata_file_path

