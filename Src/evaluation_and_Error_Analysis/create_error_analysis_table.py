import os
import pandas as pd
import json
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Define the root directory for error analysis results
root_dir = "./Data/Error_Analysis"
model_types = ["FT", "no_FT"]
models = ["distilBert", "distilRoberta", "finBert"]
model_variants = ["base", "pt", "rd_pt"]
datasets = ["eval_75_consent", "eval_all_agree"]

# Initialize an empty list to store data for CSV
data_rows = []

# Initialize a list to collect paths to confusion matrix and wordcloud images
image_paths = []

for model_type in model_types:
    for model in models:
        for variant in model_variants:
            for dataset in datasets:
                # Path construction for files
                base_path = os.path.join(root_dir, model_type, model, variant)

                # Load classification report
                report_path = os.path.join(base_path, f"classification_report_{dataset}.txt")
                if os.path.exists(report_path):
                    with open(report_path, "r") as file:
                        report = file.read()
                        lines = report.splitlines()
                        accuracy = float(lines[-3].split()[-1])
                        metrics = {}
                        for line in lines[2:-5]:  # parse precision, recall, f1-score for each class
                            parts = line.split()
                            if len(parts) >= 4:
                                label = parts[0]
                                metrics[f"{label}_precision"] = float(parts[1])
                                metrics[f"{label}_recall"] = float(parts[2])
                                metrics[f"{label}_f1_score"] = float(parts[3])
                        metrics["accuracy"] = accuracy
                        # Append results
                        data_rows.append({
                            "Model Type": model_type, "Model": model, "Variant": variant, "Dataset": dataset, **metrics
                        })

                # Collect paths for images to combine later
                confusion_matrix_path = os.path.join(base_path, f"confusion_matrix_{dataset}.png")
                wordcloud_path = os.path.join(base_path, f"misclassified_wordcloud_{dataset}.png")
                if os.path.exists(confusion_matrix_path):
                    image_paths.append((confusion_matrix_path, f"{model_type}_{model}_{variant}_{dataset}_confusion"))
                if os.path.exists(wordcloud_path):
                    image_paths.append((wordcloud_path, f"{model_type}_{model}_{variant}_{dataset}_wordcloud"))

# Save all metrics to a CSV file
df = pd.DataFrame(data_rows)
df.to_csv("error_analysis_summary.csv", index=False)
print("Saved metrics to error_analysis_summary.csv")

# Combine all images into a single comparison image
image_files = [Image.open(path) for path, _ in image_paths]
n_cols = 2  # Adjust for desired layout
n_rows = (len(image_files) + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
for i, (img, title) in enumerate(zip(image_files, [title for _, title in image_paths])):
    ax = axes[i // n_cols, i % n_cols]
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("combined_error_analysis.png")
print("Saved combined images to combined_error_analysis.png")
