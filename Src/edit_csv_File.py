import pandas as pd
import json

# Load the CSV file
file_path = 'Data/Aspect based Sentiment Analysis for Financial News.csv'
df = pd.read_csv(file_path)

# Print the columns to verify their names
print(df.columns)

# Adjust the column names based on the inspection
title_col = 'Title'  # Adjust if necessary
decision_col = 'Decisions'  # Adjust if necessary

# Keep only the Title and Decision columns and rename them to text and label
df = df[[title_col, decision_col]]
df.rename(columns={title_col: 'text', decision_col: 'label'}, inplace=True)

# Define the mapping for sentiments
sentiment_map = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

# Function to process the label column
def process_label(label):
    try:
        label_dict = json.loads(label.replace("'", '"'))
        values = list(label_dict.values())
        if all(sentiment == values[0] for sentiment in values):
            return sentiment_map[values[0]]
        else:
            return None
    except json.JSONDecodeError:
        return None

# Apply the function to the label column
df['label'] = df['label'].apply(process_label)
df['label'] = df['label'].astype(int)

# Drop rows with None in the label column
df.dropna(subset=['label'], inplace=True)

# Save the edited DataFrame to a new CSV file
output_file_path = 'Data/Processed_Financial_News.csv'
df.to_csv(output_file_path, index=False)

# Print the head of the edited DataFrame to verify
print(df.head())
