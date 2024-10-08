import pandas as pd

# Load the Parquet file
df_parquet = pd.read_parquet('/cs_storage/orkados/FinGPT_dataset.parquet')

# Drop 'instructions' column if it exists
if 'instruction' in df_parquet.columns:
    df_parquet = df_parquet.drop(columns=['instruction'])

# Function to simplify 'output' values to 'negative', 'neutral', or 'positive'
def simplify_output(value):
    value = value.lower()
    if 'positive' in value:
        return 'positive'
    elif 'negative' in value:
        return 'negative'
    else:
        return 'neutral'

# Apply the function to the 'output' column
if 'output' in df_parquet.columns:
    df_parquet['output'] = df_parquet['output'].apply(simplify_output)

df_parquet.to_csv('FinGPT_cleaned_dataset.csv', index=False)