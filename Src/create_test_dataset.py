import pandas as pd

# creates out of the 10 datasets a single dataset with the texts as raws and the cols as the labels ans saves it as 'Data/test_datasets/labeled_test_dataset.csv'
def concatenate_datasets():

    # Define the file paths for your 10 datasets
    file_paths = [f'Data/test_datasets/Financial Sentiment Labeling {i}.csv' for i in range(1, 11)]

    # List to store all processed dataframes
    all_datasets = []

    # Load and process each dataset
    for file_path in file_paths:
        try:
            # Load each dataset
            data = pd.read_csv(file_path, header=None)

            print(f'{file_path} : {data.head(3)}')

            # Extract the text (first row) and labels (next 4 rows)
            texts = data.iloc[0, :].reset_index(drop=True)
            labels = data.iloc[1:5, :].T.reset_index(drop=True)  # Transpose labels to match texts

            # Combine texts and labels into a single DataFrame
            combined_df = pd.concat([texts, labels], axis=1)
            combined_df.columns = ['text', 'labeler_1', 'labeler_2', 'labeler_3', 'labeler_4']

            # Append to the list
            all_datasets.append(combined_df)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Concatenate all datasets into one
    if all_datasets:
        final_dataset = pd.concat(all_datasets, ignore_index=True)

        # Display the first few rows to verify the structure
        print(final_dataset.head())

        # Save the final dataset to a CSV file
        final_dataset.to_csv('Data/test_datasets/labeled_test_dataset.csv', index=False)
    else:
        print("No datasets were loaded successfully.")


# creates 2 datasets, with the samples and labels with 75% consent and all agree.
def create_consensus_datasets(df):
    # Create a mapping for the labels
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

    # Map the label columns to numeric values
    df[['labeler_1', 'labeler_2', 'labeler_3', 'labeler_4']] = df[
        ['labeler_1', 'labeler_2', 'labeler_3', 'labeler_4']].replace(label_mapping)

    # Dataset with 66% or higher agreement (at least 3 out of 4 agree)
    df['agreement'] = df[['labeler_1', 'labeler_2', 'labeler_3', 'labeler_4']].mode(axis=1)[0]  # The most common label
    df['consensus_count'] = df[['labeler_1', 'labeler_2', 'labeler_3', 'labeler_4']].apply(
        lambda x: (x == x.mode()[0]).sum(), axis=1)

    # Dataset where all 4 agree
    all_agree_df = df[df['consensus_count'] == 4][['text', 'agreement']]

    # Dataset where at least 3 agree (75% and above)
    consent_75_df = df[df['consensus_count'] >= 3][['text', 'agreement']]

    # Rename 'agreement' column to 'label'
    all_agree_df = all_agree_df.rename(columns={'agreement': 'label'})
    consent_75_df = consent_75_df.rename(columns={'agreement': 'label'})

    return all_agree_df, consent_75_df

concatenate_datasets()

# Load the dataset
df = pd.read_csv('Data/test_datasets/labeled_test_dataset.csv')

# Generate the datasets based on consensus
all_agree_df, consent_75_df = create_consensus_datasets(df)

# Save the results to CSV files
all_agree_df.to_csv('Data/test_datasets/all_agree_dataset.csv', index=False)
consent_75_df.to_csv('Data/test_datasets/consent_75_dataset.csv', index=False)
