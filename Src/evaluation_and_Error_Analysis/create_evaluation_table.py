import os
import json
import pandas as pd

# Root directory for evaluation files
root_dir = "/cs_storage/orkados/Evaluation_results"

# Define the model names and directory types
models = [
    "distilroberta-finetuned-financial-news-sentiment-analysis",
    "Finbert",
    "stock-news-distilbert"
]
directories = ["FT_UPDATED", "no_FT_UPDATED"]

# Define the dataset files to prioritize "eval_75_consent" at the top
dataset_files = ["eval_75_consent.txt", "eval_all_agree.txt"]

# Initialize an empty list to collect rows for the table
rows = []

# Iterate through each directory type (FT_UPDATED, no_FT_UPDATED)
for directory in directories:
    dir_path = os.path.join(root_dir, directory)

    # Check if the directory path exists
    if os.path.isdir(dir_path):
        # Iterate through each model
        for model_name in models:
            model_dir = os.path.join(dir_path, model_name)

            # Check if the model directory exists
            if os.path.isdir(model_dir):
                # Iterate through each model type (base, pt, rd_pt)
                for model_type in ["base", "pt", "rd_pt"]:
                    model_type_dir = os.path.join(model_dir, model_type)

                    # Ensure the model_type directory exists
                    if os.path.isdir(model_type_dir):
                        # Iterate through each dataset file, ensuring "eval_75_consent" is processed first
                        for dataset_file in dataset_files:
                            file_path = os.path.join(model_type_dir, dataset_file)

                            # Check if the file exists
                            if os.path.exists(file_path):
                                with open(file_path, "r") as f:
                                    data = json.load(f)

                                    # Extract relevant metrics from JSON
                                    loss = data["results"].get("eval_loss", None)
                                    accuracy = data["results"].get("eval_accuracy", None)
                                    precision = data["results"].get("eval_precision", None)
                                    recall = data["results"].get("eval_recall", None)
                                    f1_score = data["results"].get("eval_f1", None)

                                    # Append extracted data to the list as a dictionary
                                    rows.append({
                                        "Model Directory": directory,
                                        "Model Name": model_name,
                                        "Model Type": model_type,
                                        "Dataset": dataset_file.replace(".txt", ""),  # Remove .txt extension
                                        "Loss": loss,
                                        "Accuracy": accuracy,
                                        "Precision": precision,
                                        "Recall": recall,
                                        "F1 Score": f1_score
                                    })

# Convert the collected data to a DataFrame
df = pd.DataFrame(rows)

# Sort the DataFrame to ensure "eval_75_consent" entries are listed at the top for each model and directory
df["Dataset"] = pd.Categorical(df["Dataset"], categories=["eval_75_consent", "eval_all_agree"], ordered=True)
df = df.sort_values(["Model Directory", "Model Name", "Dataset", "Model Type"]).reset_index(drop=True)

# Save the DataFrame as a CSV file
output_csv = "evaluation_results_summary_all_models.csv"
df.to_csv(output_csv, index=False)

print(f"Organized summary table for all models saved to {output_csv}")
