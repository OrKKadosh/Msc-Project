import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
all_agree_df = pd.read_csv("Data/test_datasets/all_agree_dataset.csv")
consent_75_df = pd.read_csv("Data/test_datasets/consent_75_dataset.csv")

def stratified_split(df, test_size=0.4, random_state=1694):
    # Split while preserving class distribution
    train, eval = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    return train, eval

# Split consent_75 dataset
consent_75_test, consent_75_eval = stratified_split(consent_75_df)

# Split all_agree dataset
all_agree_test, all_agree_eval = stratified_split(all_agree_df)

# Optional: Check class distribution in splits to ensure stratification
print("Consent 75 Test Distribution:\n", consent_75_test['label'].value_counts())
print("Consent 75 Eval Distribution:\n", consent_75_eval['label'].value_counts())
print("All Agree Test Distribution:\n", all_agree_test['label'].value_counts())
print("All Agree Eval Distribution:\n", all_agree_eval['label'].value_counts())

# Saving to CSV if needed
consent_75_test.to_csv("Data/test_datasets/consent_75_test.csv", index=False)
consent_75_eval.to_csv("Data/test_datasets/consent_75_eval.csv", index=False)
all_agree_test.to_csv("Data/test_datasets/all_agree_test.csv", index=False)
all_agree_eval.to_csv("Data/test_datasets/all_agree_eval.csv", index=False)