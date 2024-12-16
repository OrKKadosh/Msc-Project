# RUNS ON BOTH FILES: eval_all_agree & eval_75_consent
import time

from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import shutil
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, \
    DataCollatorWithPadding, Trainer, TrainingArguments
import evaluate

now = datetime.now()
now = now.strftime("%Y-%m-%d %H:%M:%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_DATASETS = 5
NUM_TRAIN_EPOCH = 3

def error_analysis(model_name, model_type,data, eval_dataset,eval_dataset_name):
    save_directory = f'Data/Error_Analysis/FT/{now}/{model_name}/{model_type}/{eval_dataset_name}'
    os.makedirs(save_directory, exist_ok=True)

    print(f"Starts Error_Analysis on {model_name} of type: {model_type} on {eval_dataset_name}")

    probs = np.max(data['logits'], axis=1)  # Get the max probability for each prediction

    eval_df = pd.DataFrame({
        'text': eval_dataset['text'],
        'true_label': eval_dataset['label'],
        'predicted_label': data['predictions'],
        'confidence' : probs,
    })

    eval_df['text_length'] = eval_df['text'].apply(lambda x: len(x.split()))
    # ___________to delete
    misclassified_neg_to_pos = eval_df[(eval_df['true_label'] == 0) & (eval_df['predicted_label'] == 2)]
    misclassified_text = " ".join(misclassified_neg_to_pos['text'].values)
    wordcloud = WordCloud(width=800, height=400).generate(misclassified_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Save the plot as a file
    plt.savefig(f'{save_directory}/misclassified_wordcloud_neg_to_pos.png', format="png", dpi=300)
    plt.show()
    # ___________to delete
    print("didnt go out on the return :(")
    misclassified_df = eval_df[eval_df['true_label'] != eval_df['predicted_label']]
    misclassified_df.to_csv(f'{save_directory}/misclassified.csv', index=False)

    # ___________________________Big Errors___________________________
    misclassified_pos_to_neg = eval_df[(eval_df['true_label'] == 2) & (eval_df['predicted_label'] == 0)]
    misclassified_pos_to_neg.to_csv(f'{save_directory}/misclassified_pos_to_neg.csv', index=False)

    misclassified_neg_to_pos = eval_df[(eval_df['true_label'] == 0) & (eval_df['predicted_label'] == 2)]
    misclassified_neg_to_pos.to_csv(f'{save_directory}/misclassified_neg_to_pos.csv', index=False)

    big_errors_path =  f'{save_directory}/misclassified_big_errors_report.txt'

    with open(big_errors_path, 'w') as file:
        file.write(f"Number of errors where the true_label is 2 and the predicted_label is 0 : {len(misclassified_pos_to_neg)} \n")
        file.write(f"Number of errors where the true_label is 0 and the predicted_label is 2 : {len(misclassified_neg_to_pos)}")


    # ___________________________Confusion Matrix___________________________
    cm = confusion_matrix(eval_df['true_label'], eval_df['predicted_label'], labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "neutral", "positive"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(f"{save_directory}/confusion_matrix.png")  # Save as an image
    plt.show()  # Optionally display the plot


    # ___________________________Classification report___________________________
    report = classification_report(eval_df['true_label'], eval_df['predicted_label'],target_names=["negative", "neutral", "positive"])
    print(report)  # Display report in console
    with open(f"{save_directory}/classification_report.txt","w") as file:
        file.write(report)


    # ___________________________FP & FN___________________________
    false_positives = eval_df[(eval_df['true_label'] != 0) & (eval_df['predicted_label'] == 0)]  # Model predicts negative, but should be neutral/positive
    false_positives.to_csv(f'{save_directory}/false_positives.csv', index=False)

    false_negatives = eval_df[(eval_df['true_label'] == 0) & (eval_df['predicted_label'] != 0)]  # Model fails to predict negative
    false_negatives.to_csv(f'{save_directory}/false_negatives.csv', index=False)


    # ___________________________Word cloud for misclassified texts___________________________
    misclassified_text = " ".join(misclassified_df['text'].values)
    wordcloud = WordCloud(width=800, height=400).generate(misclassified_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Save the plot as a file
    plt.savefig(f'{save_directory}/misclassified_wordcloud.png', format="png", dpi=300)
    plt.show()


    # ________________________________Word cloud for misclassified neg to post texts_________________________________________
    misclassified_neg_to_pos = eval_df[(eval_df['true_label'] == 0) & (eval_df['predicted_label'] == 2)]
    misclassified_text = " ".join(misclassified_neg_to_pos['text'].values)
    wordcloud = WordCloud(width=800, height=400).generate(misclassified_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Save the plot as a file
    plt.savefig(f'{save_directory}/misclassified_wordcloud_neg_to_pos.png', format="png", dpi=300)
    plt.show()

    # ___________________________Low confidence misclassification___________________________
    # Calculate the 25th percentile of the confidence scores
    low_confidence_threshold = np.percentile(eval_df['confidence'], 25)
    high_confidence_threshold = np.percentile(eval_df['confidence'], 75)

    low_confidence_misclassifications = eval_df[(eval_df['true_label'] != eval_df['predicted_label']) & (eval_df['confidence'] < low_confidence_threshold)]
    low_confidence_misclassifications['Low_confidence_threshold'] = low_confidence_threshold
    high_confidence_misclassifications = eval_df[(eval_df['true_label'] != eval_df['predicted_label']) & (eval_df['confidence'] >= high_confidence_threshold)]
    high_confidence_misclassifications['High_confidence_threshold'] = high_confidence_threshold

    low_confidence_misclassifications.to_csv(f'{save_directory}/low_confidence_misclassifications.csv',index=False)
    high_confidence_misclassifications.to_csv(f'{save_directory}/high_confidence_misclassifications.csv', index=False)

    avg_confidence = eval_df['confidence'].sum() / len(eval_df)
    with open(f'Data/Error_Analysis/FT/{now}/{model_name}/{model_type}/{eval_dataset_name}/confidence_average.txt', 'w') as file:
        file.write(f'The Average confidence score for this dataset : {avg_confidence}')


    # ___________________________True Neutral to Predicted Positive___________________________
    misclassified_neu_to_pos = eval_df[(eval_df['true_label'] == 1) & (eval_df['predicted_label'] == 2)]
    misclassified_neu_to_pos.to_csv(f'Data/Error_Analysis/FT/{now}/{model_name}/{model_type}/{eval_dataset_name}/misclassified_neu_to_pos.csv', index=False)

    # ___________________________Text Length___________________________
    length_save_directory = f'Data/Error_Analysis/FT/{now}/by_text_length/{model_name}/{model_type}/{eval_dataset_name}'
    os.makedirs(length_save_directory, exist_ok=True)

    # Calculate the 33rd and 66th percentiles (quantiles) for text length
    short_threshold = eval_df['text_length'].quantile(0.33)
    medium_threshold = eval_df['text_length'].quantile(0.66)

    # Classify errors by text length categories based on calculated thresholds
    short_text_errors = eval_df[(eval_df['text_length'] <= short_threshold) & (eval_df['true_label'] != eval_df['predicted_label'])]
    medium_text_errors = eval_df[(eval_df['text_length'] > short_threshold) & (eval_df['text_length'] <= medium_threshold) & (eval_df['true_label'] != eval_df['predicted_label'])]
    long_text_errors = eval_df[(eval_df['text_length'] > medium_threshold) & (eval_df['true_label'] != eval_df['predicted_label'])]

    # Save the errors in separate files
    short_text_errors.to_csv(f'{length_save_directory}/short_length_errors.csv', index=False)
    medium_text_errors.to_csv(f'{length_save_directory}/median_length_errors.csv', index=False)
    long_text_errors.to_csv(f'{length_save_directory}/long_length_errors.csv', index=False)

    # Calculate the total number of examples for each text length category
    short_text_total = eval_df[eval_df['text_length'] <= 61]
    median_text_total = eval_df[(eval_df['text_length'] <= 68) & (eval_df['text_length'] > 61)]
    long_text_total = eval_df[eval_df['text_length'] > 68]

    # Calculate error rates for each length category
    short_error_rate = len(short_text_errors) / len(short_text_total) * 100 if len(short_text_total) > 0 else 0
    median_error_rate = len(medium_text_errors) / len(median_text_total) * 100 if len(median_text_total) > 0 else 0
    long_error_rate = len(long_text_errors) / len(long_text_total) * 100 if len(long_text_total) > 0 else 0

    # Define the path for saving the error rate summary
    error_rate_summary_path = f'{length_save_directory}/error_rate_summary.txt'

    # Write the error rates to the text file
    with open(error_rate_summary_path, 'w') as file:
        file.write(f"Error Rate Summary for {eval_dataset_name}:\n")
        file.write(f"Short text (Length <= {short_threshold} words) error rate: {short_error_rate:.2f}%\n")
        file.write(f"Median text ({short_threshold} < Length <= {medium_threshold} words) error rate: {median_error_rate:.2f}%\n")
        file.write(f"Long text (Length > {medium_threshold} words) error rate: {long_error_rate:.2f}%\n")
    print(f"Error rate summary saved to {error_rate_summary_path}")

    # ________________________________________________________________

def convert_labels_to_int(example):
    # Convert the labels to integers
    example['label'] = int(example['label'])
    return example


predictions_dict = {
    'distilroberta': [
        {'type': 'base', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'rd_pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}}
    ],
    'distilbert': [
        {'type': 'base', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'rd_pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}}
    ],
    'finbert': [
        {'type': 'base', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'rd_pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}}
    ],
    'electra': [
        {'type': 'base', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'rd_pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}}
    ]
}

def tokenize_function(tokenizer, examples):
    output = tokenizer(examples['text'], padding=True, truncation=True, max_length=512)
    return output

accuracy_metric = evaluate.load("./local_metrics/accuracy")
precision_metric = evaluate.load("./local_metrics/precision")
recall_metric = evaluate.load("./local_metrics/recall")
f1_metric = evaluate.load("./local_metrics/f1")

def compute_metrics_and_save_preds(eval_pred, model_name, model_type):
    print(f"compute_metrics_and_save_preds has been called with {model_name} type: {model_type}" )
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    if model_name == 'finbert':
        id2label = {0: 2, 1: 0, 2: 1}
        predictions = [id2label[pred] for pred in predictions]

    # # Determine the main model name and access the correct dictionary based on model_type
    # if 'distilroberta' in model_name.lower():
    #     model_key = 'distilRoberta'
    # elif 'distilbert' in model_name.lower():
    #     model_key = 'distilBert'
    # elif 'finbert' in model_name.lower():
    #     model_key = 'finBert'
    # elif 'electra' in model_name.lower():
    #     model_key = 'Electra'
    # else:
    #     raise ValueError("Model name does not match any expected keys")

    # Locate the right nested dictionary by model type and store predictions, labels, and logits
    for model in predictions_dict[model_name]:
        if model['type'] == model_type:
            model['preds']['predictions'] = predictions
            model['preds']['labels'] = labels
            model['preds']['logits'] = logits
            break

    # needed while using the FINBERT & base_stock-news-distilbert, since its labels are not matching
    if model_name == 'finbert':
        id2label = {0: 2, 1: 0, 2: 1}
        mapped_predictions = [id2label[pred] for pred in predictions]
    elif model_name == 'distilbert':
        id2label = {0: 1, 1: 0, 2: 2}
        mapped_predictions = [id2label[pred] for pred in predictions]
    else:
        mapped_predictions = predictions


    # Compute accuracy, precision, recall, and f1 using either mapped or original predictions
    accuracy = accuracy_metric.compute(predictions=mapped_predictions, references=labels)
    precision = precision_metric.compute(predictions=mapped_predictions, references=labels, average='macro')
    recall = recall_metric.compute(predictions=mapped_predictions, references=labels, average='macro')
    f1 = f1_metric.compute(predictions=mapped_predictions, references=labels, average='macro')

    return {
      'accuracy': accuracy['accuracy'],
      'precision': precision['precision'],
      'recall': recall['recall'],
      'f1': f1['f1']
    }

# returns the shortcut name
def get_model_name(model_name):
    if 'distilroberta' in model_name.lower():
        model_key = 'distilRoberta'
    elif 'distilbert' in model_name.lower():
        model_key = 'distilBert'
    elif 'finbert' in model_name.lower():
        model_key = 'finBert'
    elif 'electra' in model_name.lower():
        model_key = 'Electra'
    else:
        raise ValueError("Model name does not match any expected keys")
    return model_key

eval_consent_75_df = pd.read_csv('Data/test_datasets/split_eval_test/consent_75_eval.csv').apply(convert_labels_to_int, axis=1)
eval_consent_75 = Dataset.from_pandas(eval_consent_75_df)

eval_all_agree_df = pd.read_csv('Data/test_datasets/split_eval_test/all_agree_eval.csv').apply(convert_labels_to_int, axis=1)
eval_all_agree = Dataset.from_pandas(eval_all_agree_df)

models_names = ['distilroberta', 'distilbert', 'finbert', 'electra']
models_types = ['base', 'pt', 'rd_pt']
eval_datasets = [{'dataset': eval_all_agree, 'name': 'eval_all_agree'}, {'dataset': eval_consent_75, 'name': 'eval_consent_75'}]


def perform_error_analysis(eval_dataset):

    # Set up evaluation arguments
    evaluation_args = TrainingArguments(
        output_dir="./eval_checkpoints",
        per_device_eval_batch_size=2,
        logging_dir='./logs',
        do_eval=True,
        save_strategy="epoch",
    )

    eval_results = []  # To store each evaluation result, including dataset, dataset name, and predictions
    # __________________________
    for model_name in models_names:
        for model_type in models_types:

            # Load the Fine-Tuned model for evaluation
            save_directory = f'./Saved_models/fine-tuned/{model_name}_{model_type}'
            model = AutoModelForSequenceClassification.from_pretrained(save_directory, num_labels=3).to(device)
            tokenizer = AutoTokenizer.from_pretrained(save_directory)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

            print(f"Starts evaluating the FT model: {model_name} of type: {model_type} on dataset: {eval_dataset['name']}")

            tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(tokenizer, x),batched=True)

            # Initialize the Trainer for the evaluation phase
            trainer = Trainer(
                model=model,
                args=evaluation_args,
                eval_dataset=tokenized_eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=lambda eval_pred: compute_metrics_and_save_preds(eval_pred, model_name, model_type),
            )

            evaluation_results = trainer.evaluate()

            results_with_model = {
                "Type": model_type,
                "model_name": model_name,
                "results": evaluation_results,
                "eval_args": evaluation_args.to_dict(),
            }

            results_file_name = f"{eval_dataset['name']}.txt"
            results_dir = f"./Evaluation_results/FT_{eval_dataset['name']}/{model_name}/{model_type}/"
            os.makedirs(results_dir, exist_ok=True)
            results_file_path = os.path.join(results_dir, results_file_name)

            with open(results_file_path, "w") as file:
                file.write(json.dumps(results_with_model, indent=4))

            print(f"Evaluation results for the FT model: {model_name} of type: {model_type} saved to {results_file_name}")

            # Save evaluation results and data for error analysis
            eval_results.append({
                'model_name': model_name,
                'model_type': model_type,
                'eval_dataset': eval_dataset['dataset'],
                'eval_dataset_name': eval_dataset['name'],
                'evaluation_results': evaluation_results
            })

    # Run error analysis on the collected results
    for result in eval_results:
        model_name = result['model_name']
        model_type = result['model_type']
        eval_dataset = result['eval_dataset']
        eval_dataset_name = result['eval_dataset_name']

        # Access the correct prediction data
        for type_data in predictions_dict[model_name]:
            if type_data['type'] == model_type:
                data = type_data['preds']
                error_analysis(model_name, model_type, data, eval_dataset, eval_dataset_name)
                print(f"ended with {model_name} of type {type_data['type']}")
                break

def perform_error_analysis_calc(eval_dataset):
    """
    Perform error analysis by evaluating models, collecting predictions, and calculating statistics.
    """

    # Set up evaluation arguments
    evaluation_args = TrainingArguments(
        output_dir="./eval_checkpoints",
        per_device_eval_batch_size=2,
        logging_dir='./logs',
        do_eval=True,
        save_strategy="epoch",
    )

    eval_results = []  # To store evaluation results and predictions

    for model_name in models_names:
        for model_type in models_types:

            # Load the Fine-Tuned model for evaluation
            save_directory = f'./Saved_models/fine-tuned/{model_name}_{model_type}'
            model = AutoModelForSequenceClassification.from_pretrained(save_directory, num_labels=3).to(device)
            tokenizer = AutoTokenizer.from_pretrained(save_directory)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

            print(f"Starts evaluating the FT model: {model_name} of type: {model_type} on dataset: {eval_dataset['name']}")

            tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(tokenizer, x), batched=True)

            # Add an index column to track original row indices
            eval_dataset_with_index = eval_dataset['dataset'].map(
                lambda x, idx: {"index": idx},
                with_indices=True
            )

            # Initialize the Trainer for the evaluation phase
            trainer = Trainer(
                model=model,
                args=evaluation_args,
                eval_dataset=tokenized_eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=lambda eval_pred: compute_metrics_and_save_preds(eval_pred, model_name, model_type),
            )

            # Perform evaluation and get predictions
            predictions = trainer.predict(tokenized_eval_dataset)
            predicted_labels = predictions.predictions.argmax(axis=1)

            # Filter samples containing the word 'dividend'
            word = 'dividend'
            word_occurrences = eval_dataset_with_index.filter(
                lambda x: word.lower() in x['text'].lower()
            )

            # Align predictions with the filtered dataset using the index
            filtered_indices = word_occurrences['index']
            word_occurrences = word_occurrences.add_column(
                "predicted_label", [predicted_labels[idx] for idx in filtered_indices]
            )

            # Calculate statistics
            total_counts = {label: (predicted_labels == label).sum() for label in set(predicted_labels)}
            word_counts = {label: sum(1 for pred in word_occurrences['predicted_label'] if pred == label)
                           for label in set(predicted_labels)}

            percentages = {
                label: (word_counts[label] / total_counts[label] * 100) if label in word_counts else 0
                for label in total_counts.keys()
            }

            # Save results to a text file
            stats_dir = f"./Error_Analysis_Stats/{model_name}/{model_type}"
            os.makedirs(stats_dir, exist_ok=True)
            stats_file_path = os.path.join(stats_dir, "word_statistics.txt")

            with open(stats_file_path, "w") as file:
                file.write(f"Model: {model_name}, Type: {model_type}\n")
                file.write(f"Word: '{word}' Statistics by Predicted Label\n")
                file.write("=======================================\n")
                for label, percentage in percentages.items():
                    file.write(f"Label {label}: {percentage:.2f}%\n")
                file.write("\n")

            print(f"Statistics saved for {model_name}, Type: {model_type}")

    print("Error analysis completed.")


def organize_classification_report():
    base_dir = f"/cs_storage/orkados/Data/Error_Analysis/FT/2024-11-11 17:16:57/sorted_by_files/classification_report"
    output_file_path = os.path.join(base_dir, "total_classification_report.txt")
    models_names = ['distilroberta', 'distilbert', 'finbert', 'electra']
    models_types = ['base', 'pt', 'rd_pt']
    eval_datasets = ['eval_all_agree', 'eval_consent_75']

    def parse_classification_report(file_path):
        metrics = {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 4 and parts[0] in ["negative", "neutral", "positive"]:
                    label = parts[0].capitalize()
                    metrics[label] = {
                        "Precision": parts[1],
                        "Recall": parts[2],
                        "F1-Score": parts[3]
                    }
                elif len(parts) == 3 and parts[0] == "accuracy":
                    metrics["Overall"] = {
                        "Precision": parts[1],
                        "Recall": parts[1],
                        "F1-Score": parts[2]
                    }
        return metrics

    with open(output_file_path, 'w') as output_file:
        output_file.write("Total Classification Report\n")
        output_file.write("=" * 80 + "\n")

        for model_name in models_names:
            output_file.write(f"Model: {model_name}\n")
            output_file.write("-" * 80 + "\n")

            for eval_dataset in eval_datasets:
                output_file.write(f"  Evaluation Dataset: {eval_dataset}\n")
                output_file.write(f"{'Label':<12} | {'Type':<14} | {'Precision':<10} | {'Recall':<8} | {'F1-Score'}\n")
                output_file.write("-" * 80 + "\n")

                for label in ["Negative", "Neutral", "Positive", "Overall"]:
                    for model_type in models_types:
                        file_name = f"{model_name}_{model_type}_classification_report.txt"
                        file_path = os.path.join(base_dir, file_name)

                        if os.path.exists(file_path):
                            metrics = parse_classification_report(file_path)
                            if label in metrics:
                                precision = metrics[label]["Precision"]
                                recall = metrics[label]["Recall"]
                                f1_score = metrics[label]["F1-Score"]
                                output_file.write(
                                    f"{label:<12} | {model_type:<14} | {precision:<10} | {recall:<8} | {f1_score}\n"
                                )
                    output_file.write("-" * 80 + "\n")
            output_file.write("=" * 80 + "\n")

    print("Total classification report generated at:", output_file_path)

# organize files by model, type, and dataset into a structured directory
def move_files():
    # Base directory containing the model folders
    base_dir = f"/cs_storage/orkados/Data/Error_Analysis/FT/{now}"
    sorted_dir = os.path.join(base_dir, "sorted_by_files")
    os.makedirs(sorted_dir, exist_ok=True)

    # Model, type, and evaluation dataset names
    models_names = ['distilroberta', 'distilbert', 'finbert', 'electra']
    models_types = ['base', 'pt', 'rd_pt']
    eval_datasets = ['eval_all_agree', 'eval_consent_75']

    # Iterate through all files in the subdirectories
    for root, _, files in os.walk(base_dir):
        if "sorted_by_files" in root:  # Skip the sorted directory itself
            continue

        # Extract model_name, model_type, and eval_dataset from the directory path if they match
        path_parts = Path(root).parts
        model_name = next((name for name in models_names if name in path_parts), None)
        model_type = next((typ for typ in models_types if typ in path_parts), None)
        eval_dataset = next((dataset for dataset in eval_datasets if dataset in path_parts), None)

        for file in files:
            file_path = os.path.join(root, file)

            # Create a folder in sorted_dir based on file type (without extension), then subfolder by dataset
            file_name = Path(file).stem  # Get filename without extension
            target_dir = os.path.join(sorted_dir, file_name, eval_dataset)
            os.makedirs(target_dir, exist_ok=True)

            # Construct a unique file name with model_name, model_type, and eval_dataset
            if model_name and model_type and eval_dataset:
                new_file_name = f"{model_name}_{model_type}_{eval_dataset}_{file}"
            else:
                new_file_name = file  # Use the original name if any attribute is not found

            # Copy the file to the target directory with the new name
            shutil.copy2(file_path, os.path.join(target_dir, new_file_name))

    print("Files sorted by name and copied to the 'sorted_by_files' folder.")

# organize classification report data from text files into a single Excel file for easier comparison across models, types, and datasets
def create_total_classification_report_to_excel():
    # Directory paths
    base_dir = f"/cs_storage/orkados/Data/Error_Analysis/FT/{now}/sorted_by_files/classification_report"
    output_file_path = os.path.join(base_dir, "total_classification_report.xlsx")

    # Model, type, and dataset names
    models_names = ['distilroberta', 'distilbert', 'finbert', 'electra']
    models_types = ['base', 'pt', 'rd_pt']
    eval_datasets = ['eval_all_agree', 'eval_consent_75']

    # Initialize an empty list to gather report data
    report_data = []

    # Parse classification report files and gather data
    for dataset in eval_datasets:
        dataset_dir = os.path.join(base_dir, dataset)  # Directory for each dataset
        for model_name in models_names:
            for model_type in models_types:
                file_name = f"{model_name}_{model_type}_{dataset}_classification_report.txt"
                file_path = os.path.join(dataset_dir, file_name)

                # Check if the file exists before proceeding
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue

                # Parse the classification report file and add its data
                with open(file_path, 'r') as file:
                    for line in file:
                        # Parsing lines for individual labels
                        parts = line.strip().split()
                        if len(parts) == 5 and parts[0].lower() in ["negative", "neutral", "positive"]:
                            label = parts[0].capitalize()
                            precision, recall, f1_score = parts[1], parts[2], parts[3]
                            report_data.append({
                                "Model": model_name,
                                "Type": model_type,
                                "Dataset": dataset,
                                "Label": label,
                                "Precision": precision,
                                "Recall": recall,
                                "F1-Score": f1_score
                            })
                        # Parsing the macro avg and weighted avg rows
                        elif len(parts) >= 4 and parts[0] in ["macro", "weighted"] and parts[1] == "avg":
                            label = f"{parts[0].capitalize()} Avg"
                            precision, recall, f1_score = parts[2], parts[3], parts[4]
                            report_data.append({
                                "Model": model_name,
                                "Type": model_type,
                                "Dataset": dataset,
                                "Label": label,
                                "Precision": precision,
                                "Recall": recall,
                                "F1-Score": f1_score
                            })

    # Convert gathered data to a DataFrame and save as Excel file
    df = pd.DataFrame(report_data)
    df.to_excel(output_file_path, index=False)

    print("Total classification report saved to:", output_file_path)

# create a collage of confusion matrix images for each model, organized by model type and evaluation dataset, with titles above each image
def create_total_confusion_matrix_collage():
    base_dir = f"/cs_storage/orkados/Data/Error_Analysis/FT/{now}/sorted_by_files"
    output_base_path = os.path.join(base_dir, "confusion_matrix")
    os.makedirs(output_base_path, exist_ok=True)

    models_names = ['distilroberta', 'distilbert', 'finbert', 'electra']
    models_types = ['base', 'pt', 'rd_pt']
    eval_datasets = ['eval_all_agree', 'eval_consent_75']

    # Set font size for titles
    font_size = 24  # Adjusted for better readability
    padding = 20
    title_height = 40  # Height reserved for each title text

    # Try loading a larger font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Use a truetype font if available
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if not available

    # Generate a collage for each model
    for model_name in models_names:
        images_info = []

        # Gather confusion matrix images for the current model with appropriate titles
        for dataset in eval_datasets:
            dataset_dir = os.path.join(base_dir, "confusion_matrix", dataset)  # Directory for each dataset
            for model_type in models_types:
                filename = f"{model_name}_{model_type}_{dataset}_confusion_matrix.png"
                file_path = os.path.join(dataset_dir, filename)
                if os.path.exists(file_path):
                    title = f"{model_name}_{model_type}_{dataset}"
                    images_info.append((title, Image.open(file_path)))
                else:
                    print(f"File not found: {file_path}")

        if not images_info:
            print(f"No confusion matrix images found for model {model_name}.")
            continue

        # Layout configurations based on image dimensions
        img_width, img_height = images_info[0][1].size
        num_columns = len(models_types)  # Number of columns per row
        num_rows = len(eval_datasets)

        # Calculate final image dimensions
        final_width = img_width * num_columns + padding * (num_columns + 1)
        final_height = (img_height + title_height + padding) * num_rows + padding

        # Create the final image canvas for this model
        collage_image = Image.new("RGB", (final_width, final_height), "white")
        draw = ImageDraw.Draw(collage_image)

        # Arrange images by row and column
        for row_idx, dataset in enumerate(eval_datasets):
            y_offset = row_idx * (img_height + title_height + padding) + padding
            for col_idx, (title, img) in enumerate([info for info in images_info if dataset in info[0]]):
                x_offset = col_idx * (img_width + padding) + padding

                # Draw the image
                collage_image.paste(img, (x_offset, y_offset + title_height))

                # Draw the title centered above the image
                text_bbox = draw.textbbox((0, 0), title, font=font)
                text_width = text_bbox[2] - text_bbox[0]  # Width of the text
                text_x = x_offset + (img_width - text_width) // 2
                text_y = y_offset
                draw.text((text_x, text_y), title, font=font, fill="black")

        # Save the final collage image for this model
        output_path = os.path.join(output_base_path, f"total_confusion_matrix_{model_name}.png")
        collage_image.save(output_path)
        print(f"Confusion matrix collage for {model_name} saved to {output_path}")

def create_total_big_errors_report():
    # Base directories
    base_dir = f"/cs_storage/orkados/Data/Error_Analysis/FT/{now}/sorted_by_files"
    report_dir = os.path.join(base_dir, "misclassified_big_errors_report")
    output_dir = os.path.join(report_dir, "excel_reports")
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters
    eval_datasets = ["eval_all_agree", "eval_consent_75"]
    models_names = ["distilroberta", "distilbert", "finbert", "electra"]
    models_types = ["base", "pt", "rd_pt"]

    # Loop through each eval_dataset to create separate Excel files
    for eval_dataset in eval_datasets:
        report_data = []  # List to store rows of data for the Excel file

        # Gather report data for each model and type
        for model_name in models_names:
            for model_type in models_types:
                filename = f"{model_name}_{model_type}_{eval_dataset}_misclassified_big_errors_report.txt"
                file_path = os.path.join(report_dir, eval_dataset, filename)

                if os.path.exists(file_path):
                    with open(file_path, "r") as file:
                        errors = {"Model": model_name, "Type": model_type, "Dataset": eval_dataset}
                        for line in file:
                            if "true_label is 2 and the predicted_label is 0" in line:
                                errors["True 2, Pred 0"] = int(line.split(":")[1].strip())
                            elif "true_label is 0 and the predicted_label is 2" in line:
                                errors["True 0, Pred 2"] = int(line.split(":")[1].strip())
                        report_data.append(errors)
                else:
                    # If file is missing, add zeros for error counts
                    report_data.append({
                        "Model": model_name,
                        "Type": model_type,
                        "Dataset": eval_dataset,
                        "True 2, Pred 0": 0,
                        "True 0, Pred 2": 0
                    })

        # Create a DataFrame and save it as an Excel file
        df = pd.DataFrame(report_data)
        output_path = os.path.join(output_dir, f"{eval_dataset}_misclassified_big_errors_comparison.xlsx")
        df.to_excel(output_path, index=False)
        print(f"Excel report for {eval_dataset} saved to {output_path}")

def create_total_wordcloud_collage():
    base_dir = f"/cs_storage/orkados/Data/Error_Analysis/FT/{now}/sorted_by_files"
    output_base_path = os.path.join(base_dir, "misclassified_wordcloud")
    os.makedirs(output_base_path, exist_ok=True)

    models_names = ['distilroberta', 'distilbert', 'finbert', 'electra']
    models_types = ['base', 'pt', 'rd_pt']
    eval_datasets = ['eval_all_agree', 'eval_consent_75']

    padding = 40  # Padding between images
    title_height_ratio = 0.25  # Titles will take up 25% of each wordcloud height for visibility

    # Try loading a bold font
    font_path = "arialbd.ttf"  # Path to a bold font file
    try:
        font = ImageFont.truetype(font_path, 50)  # Initial font size (will adjust later)
    except IOError:
        font = ImageFont.load_default()

    # Generate a collage for each dataset
    for dataset in eval_datasets:
        images_info = []

        # Gather word cloud images for the current dataset
        for model_name in models_names:
            for model_type in models_types:
                filename = f"{model_name}_{model_type}_{dataset}_misclassified_wordcloud.png"
                file_path = os.path.join(base_dir, "misclassified_wordcloud", dataset, filename)
                if os.path.exists(file_path):
                    title = f"{model_name}_{model_type}_{dataset}"
                    images_info.append((title, Image.open(file_path)))
                else:
                    print(f"File not found: {file_path}")

        if not images_info:
            print(f"No word cloud images found for dataset {dataset}.")
            continue

        # Get dimensions of a single word cloud image
        img_width, img_height = images_info[0][1].size
        title_height = int(img_height * title_height_ratio)

        # Set up collage dimensions
        num_columns = len(models_types)
        num_rows = len(models_names)
        final_width = img_width * num_columns + padding * (num_columns + 1)
        final_height = (img_height + title_height + padding) * num_rows + padding

        # Create the collage canvas
        collage_image = Image.new("RGB", (final_width, final_height), "white")

        # Paste word cloud images into the collage without titles
        y_offset = padding
        for row_idx, model_name in enumerate(models_names):
            x_offset = padding
            for col_idx, (title, img) in enumerate([info for info in images_info if model_name in info[0]]):
                collage_image.paste(img, (x_offset, y_offset + title_height))
                x_offset += img_width + padding
            y_offset += img_height + title_height + padding

        # Re-open the collage to draw titles over it
        draw = ImageDraw.Draw(collage_image)

        # Add titles on top of each image
        y_offset = padding
        for row_idx, model_name in enumerate(models_names):
            x_offset = padding
            for col_idx, (title, img) in enumerate([info for info in images_info if model_name in info[0]]):
                # Adjust font size dynamically based on title area width
                adjusted_font_size = int(img_height * title_height_ratio * 0.8)  # Adjust size relative to title height
                try:
                    font = ImageFont.truetype(font_path, adjusted_font_size)
                except IOError:
                    font = ImageFont.load_default()

                # Calculate position to center the title above the image
                text_bbox = draw.textbbox((0, 0), title, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x_offset + (img_width - text_width) // 2
                text_y = y_offset + (title_height - adjusted_font_size) // 2
                draw.text((text_x, text_y), title, font=font, fill="black")  # Draw the title in black

                x_offset += img_width + padding
            y_offset += img_height + title_height + padding

        # Save the final collage image for this dataset
        output_path = os.path.join(output_base_path, f"total_wordcloud_{dataset}.png")
        collage_image.save(output_path)
        print(f"Word cloud collage for {dataset} saved to {output_path}")
# compile average confidence scores from text files for each model, type, and dataset, saving the results to an Excel summary
def create_total_confidence_averages():
    # Directories and paths
    base_dir = f"/cs_storage/orkados/Data/Error_Analysis/FT/{now}/sorted_by_files/confidence_average"
    output_file = os.path.join(base_dir, "total_confidence_average_summary.xlsx")

    # Model, type, and dataset names
    models_names = ['distilroberta', 'distilbert', 'finbert', 'electra']
    models_types = ['base', 'pt', 'rd_pt']
    eval_datasets = ['eval_all_agree', 'eval_consent_75']

    # Initialize list to store confidence average data
    confidence_data = []

    # Gather confidence average from each text file
    for dataset in eval_datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        for model_name in models_names:
            for model_type in models_types:
                file_name = f"{model_name}_{model_type}_{dataset}_confidence_average.txt"
                file_path = os.path.join(dataset_dir, file_name)

                # Check if the file exists
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        line = file.readline()
                        # Extract the confidence score from the file
                        score = float(line.split(":")[-1].strip())
                        confidence_data.append({
                            "Model": model_name,
                            "Type": model_type,
                            "Dataset": dataset,
                            "Average Confidence Score": score
                        })
                else:
                    print(f"File not found: {file_path}")

    # Create a DataFrame and save to Excel
    df = pd.DataFrame(confidence_data)
    df.to_excel(output_file, index=False)

    print(f"Confidence average summary saved to: {output_file}")


# compile error rate summaries from text files, dynamically including text lengths in column names, and saving the results to an Excel file
def create_error_rate_summary_excel():
    base_dir = f"/cs_storage/orkados/Data/Error_Analysis/FT/{now}/sorted_by_files/error_rate_summary"
    output_file = os.path.join(base_dir, "total_error_rate_summary.xlsx")

    models_names = ['distilroberta', 'distilbert', 'finbert', 'electra']
    models_types = ['base', 'pt', 'rd_pt']
    eval_datasets = ['eval_all_agree', 'eval_consent_75']

    report_data = []

    for dataset in eval_datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        for model_name in models_names:
            for model_type in models_types:
                file_name = f"{model_name}_{model_type}_{dataset}_error_rate_summary.txt"
                file_path = os.path.join(dataset_dir, file_name)

                if not os.path.exists(file_path):
                    continue

                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    short_error_rate = float(lines[1].split(":")[1].strip().replace("%", ""))
                    median_error_rate = float(lines[2].split(":")[1].strip().replace("%", ""))
                    long_error_rate = float(lines[3].split(":")[1].strip().replace("%", ""))

                    report_data.append({
                        "Model": model_name,
                        "Type": model_type,
                        "Dataset": dataset,
                        "Short Text Error Rate (%)": short_error_rate,
                        "Median Text Error Rate (%)": median_error_rate,
                        "Long Text Error Rate (%)": long_error_rate,
                    })

    df = pd.DataFrame(report_data)
    df.to_excel(output_file, index=False)

    print(f"Total error rate summary saved to: {output_file}")


# tmp_create_error_rate_summary_excel()

from collections import defaultdict

import re
from collections import defaultdict

def calculate_true_label_statistics(dataset, word_pattern=r"\bdividen\w*\b"):
    """
    Calculate the percentages of a word or its variations in each true label for the dataset.
    """

    # Initialize dictionaries to store counts
    total_counts = defaultdict(int)
    word_counts = defaultdict(int)
    total_word_count = 0  # Total occurrences of the word across all labels

    # Compile regex pattern for efficiency
    word_regex = re.compile(word_pattern, re.IGNORECASE)

    # Iterate through the dataset
    for row in dataset:
        label = row['label']
        text = row['text']

        # Increment total count for the label
        total_counts[label] += 1

        # Increment word count if the word is in the text
        if word_regex.search(text):
            word_counts[label] += 1
            total_word_count += 1

    # Calculate percentages normalized to the total word count
    percentages = {
        label: (word_counts[label] / total_word_count * 100)
        if total_word_count > 0 else 0
        for label in total_counts
    }

    # Debugging info
    print("Total Word Occurrences:", total_word_count)
    print("Total Counts by Label:", dict(total_counts))
    print("Word Counts by Label:", dict(word_counts))
    print("Percentages:", percentages)
    print("=" * 50)

    return percentages

import re
from collections import defaultdict
import os

import re
from collections import defaultdict
import os

def calculate_predicted_label_statistics_combined(predictions_dict, eval_dataset, word_pattern=r"\bdividen\w*\b"):
    """
    Calculate the percentages of the specified word or its variations in each predicted label for all models,
    and save the stats in a single text file. Ensures consistent totals.
    """
    stats_dir = "./Prediction_Statistics"
    os.makedirs(stats_dir, exist_ok=True)
    combined_stats_file_path = os.path.join(stats_dir, "combined_word_statistics.txt")

    with open(combined_stats_file_path, "w") as combined_file:
        combined_file.write("Combined Word Statistics for All Models\n")
        combined_file.write("=======================================\n")

        for model_name, model_data in predictions_dict.items():
            for model_type_data in model_data:
                model_type = model_type_data['type']
                preds = model_type_data['preds']

                predictions = preds['predictions']
                labels = preds['labels']

                # Explicitly check if predictions are empty
                if len(predictions) == 0:
                    combined_file.write(f"Skipping {model_name} {model_type} as predictions are empty.\n")
                    continue

                # Add the predicted label to the evaluation dataset
                eval_dataset_with_preds = eval_dataset.map(
                    lambda x, idx: {'predicted_label': predictions[idx]},
                    with_indices=True
                )

                # Initialize dictionaries to store counts
                total_counts = defaultdict(int)  # Total samples per label
                word_counts = defaultdict(int)  # Word occurrences per label
                total_word_count = 0  # Total occurrences of the word across all labels

                # Compile regex pattern for efficiency
                word_regex = re.compile(word_pattern, re.IGNORECASE)

                # Iterate through the evaluation dataset with predictions
                for i in range(len(eval_dataset_with_preds)):
                    row = eval_dataset_with_preds[i]
                    label = row['predicted_label']
                    text = row['text']

                    # Increment total count for the predicted label
                    total_counts[label] += 1

                    # Increment word count if the word pattern is found in the text
                    if word_regex.search(text):
                        word_counts[label] += 1
                        total_word_count += 1

                # Calculate percentages normalized to the total word count
                percentages = {
                    label: (word_counts[label] / total_word_count * 100)
                    if total_word_count > 0 else 0
                    for label in total_counts
                }

                # Write statistics to the combined text file
                combined_file.write(f"Model: {model_name}, Type: {model_type}\n")
                combined_file.write(f"Word Pattern: '{word_pattern}' Statistics by Predicted Label\n")
                combined_file.write("=======================================\n")
                combined_file.write(f"Total Word Occurrences: {total_word_count}\n")
                for label, percentage in percentages.items():
                    combined_file.write(f"Label {label}: {percentage:.2f}%\n")
                combined_file.write("\n")

                # Debugging info
                print(f"Model: {model_name}, Type: {model_type}")
                print(f"Total Word Occurrences: {total_word_count}")
                print(f"Total Counts by Label: {dict(total_counts)}")
                print(f"Word Counts by Label: {dict(word_counts)}")
                print(f"Percentages: {percentages}")
                print("=" * 50)

    print(f"Combined statistics saved to {combined_stats_file_path}.")


# for eval_dataset in eval_datasets:
eval_dataset = {'dataset': eval_consent_75, 'name': 'eval_consent_75'}
# perform_error_analysis(eval_dataset)

# ________Calculate the distribution of the word 'dividend'_______
# percentages = calculate_true_label_statistics(eval_dataset['dataset'])
# print("Percentage of word occurrences in true labels:")
# print(percentages)
# perform_error_analysis(eval_dataset)
# calculate_predicted_label_statistics_combined(predictions_dict, eval_dataset['dataset'])
# ________Calculate the distribution of the word 'dividend'_______

# move_files()
# create_total_classification_report_to_excel()
# create_total_confusion_matrix_collage()
# create_total_confidence_averages()
# create_error_rate_summary_excel()
# create_total_big_errors_report()
# create_total_wordcloud_collage()
def clean_dataset(dataset, idx):
    if idx == 0: #fiqa
        def remove_https(text):
            url_pattern = re.compile(r'https://\S+')
            return url_pattern.sub(r'', text)
        def clean_sentence(example):
            example['text'] = remove_https(example['text'])
            return example

        cleaned_dataset = dataset.map(lambda example: clean_sentence(example))

    elif idx == 2: #stock-market sentiment dataset
        def remove_word(text):
            word_pattern = re.compile(r'\b{}\b'.format(re.escape('user')), flags=re.IGNORECASE)
            return word_pattern.sub(r'', text)

        cleaned_dataset = dataset.map(lambda example: {'text': remove_word(example['text'])})
    return cleaned_dataset

# ___________check attention of "dividend"__________
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./Saved_models/fine-tuned/finbert_base", num_labels=3).to(device)
tokenizer = AutoTokenizer.from_pretrained("./Saved_models/fine-tuned/finbert_base")
def calculate_avg_attention(target_words, dataset):
    target_token_ids = [tokenizer(word, add_special_tokens=False)["input_ids"] for word in target_words]
    target_token_ids = [torch.tensor(ids).to(device) for ids in target_token_ids]  # Move to device


    # Initialize accumulators
    total_target_attention = 0.0
    total_other_attention = 0.0
    total_target_tokens = 0
    total_other_tokens = 0

    # Process each sample in the dataset
    for sample in dataset:
        text = sample["text"]
        input_data = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = input_data["input_ids"]

        # Forward pass with attention
        outputs = model(**input_data, output_attentions=True)
        attention_weights = outputs.attentions  # (num_layers, batch_size, num_heads, seq_len, seq_len)

        # Find positions of the target tokens in the input
        positions = [
            i for i, token_id in enumerate(input_ids[0])
            if any(token_id in ids for ids in target_token_ids)
        ]

        # Aggregate attention for the sample
        sample_target_attention = 0.0
        sample_other_attention = 0.0
        seq_len = input_ids.size(1)

        for layer_attention in attention_weights:  # Iterate over layers
            for head_attention in layer_attention[0]:  # Iterate over heads
                head_attention = head_attention[0]  # Extract the attention for this batch
                if positions:
                    # Compute attention for target tokens
                    sample_target_attention += sum(head_attention[pos].sum().item() for pos in positions)

                # Compute attention for all tokens
                sample_other_attention += head_attention.sum().item()

        # Normalize attention scores
        target_attention = sample_target_attention / (len(positions) * len(attention_weights) * len(attention_weights[0]) or 1)
        other_attention = (sample_other_attention - sample_target_attention) / ((seq_len - len(positions)) * len(attention_weights) * len(attention_weights[0]) or 1)

        # Update accumulators
        total_target_attention += target_attention
        total_other_attention += other_attention
        total_target_tokens += len(positions)
        total_other_tokens += (seq_len - len(positions))

    # Compute overall averages
    avg_target_attention = total_target_attention / total_target_tokens if total_target_tokens > 0 else 0.0
    avg_other_attention = total_other_attention / total_other_tokens if total_other_tokens > 0 else 0.0

    # Log the results
    print(f"Overall Average Attention for '{target_words[0]}' and '{target_words[1]}': {avg_target_attention:.4f}")
    print(f"Overall Average Attention for Other Tokens: {avg_other_attention:.4f}")

calculate_avg_attention(["dividend", "dividends"], eval_consent_75)
calculate_avg_attention(["cut", "cuts"], eval_consent_75)
calculate_avg_attention(["share", "shares"], eval_consent_75)
calculate_avg_attention(["profit", "profits"], eval_consent_75)
calculate_avg_attention(["yield", "yields"], eval_consent_75)

# ______________________________________________
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import matplotlib.pyplot as plt
#
# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# # Load model and tokenizer
# model = AutoModelForSequenceClassification.from_pretrained("./Saved_models/fine-tuned/finbert_base", num_labels=3).to(device)
# tokenizer = AutoTokenizer.from_pretrained("./Saved_models/fine-tuned/finbert_base")
#
# # Define target words and comparison words
# target_words = ["dividend", "dividends"]
# comparison_words = ["cut", "pay", "share", "profit", "revenue", "yield", "interest"]
#
# # Tokenize the comparison words
# comparison_token_ids = [tokenizer(word, add_special_tokens=False)["input_ids"] for word in comparison_words]
#
# # Initialize accumulators for normalized scores
# total_target_attention = 0
# total_comparison_attention = 0
# total_other_attention = 0
# total_target_tokens = 0
# total_comparison_tokens = 0
# total_other_tokens = 0
#
# # Process the dataset
# for sample in eval_consent_75:  # Iterate over each sample in the dataset
#     input_data = tokenizer(sample["text"], return_tensors="pt", padding=True, truncation=True).to(device)
#     input_ids = input_data["input_ids"]
#
#     # Perform forward pass with attention
#     outputs = model(**input_data, output_attentions=True)
#     attention_weights = outputs.attentions  # (num_layers, batch_size, num_heads, seq_len, seq_len)
#
#     # Find positions for target words and comparison words
#     target_positions = [
#         i for i, token_id in enumerate(input_ids[0])
#         if any(token_id in ids for ids in [tokenizer(word, add_special_tokens=False)["input_ids"] for word in target_words])
#     ]
#     comparison_positions = [
#         i for i, token_id in enumerate(input_ids[0])
#         if any(token_id in ids for ids in comparison_token_ids)
#     ]
#
#     # Aggregate attention scores for each token group
#     for layer_attention in attention_weights:  # Iterate over layers
#         for head_attention in layer_attention[0]:  # Iterate over heads
#             # Compute target words attention
#             if target_positions:
#                 target_attention = sum(head_attention[:, pos].mean().item() for pos in target_positions)
#                 total_target_attention += target_attention
#                 total_target_tokens += len(target_positions)
#
#             # Compute comparison words attention
#             if comparison_positions:
#                 comparison_attention = sum(head_attention[:, pos].mean().item() for pos in comparison_positions)
#                 total_comparison_attention += comparison_attention
#                 total_comparison_tokens += len(comparison_positions)
#
#             # Compute other tokens attention
#             all_positions = set(range(head_attention.size(1)))
#             other_positions = all_positions - set(target_positions) - set(comparison_positions)
#             if other_positions:
#                 other_attention = sum(head_attention[:, pos].mean().item() for pos in other_positions)
#                 total_other_attention += other_attention
#                 total_other_tokens += len(other_positions)
#
# # Normalize by token counts
# avg_target_attention = total_target_attention / total_target_tokens if total_target_tokens else 0.0
# avg_comparison_attention = total_comparison_attention / total_comparison_tokens if total_comparison_tokens else 0.0
# avg_other_attention = total_other_attention / total_other_tokens if total_other_tokens else 0.0
#
# # Log the results
# print(f"Normalized Average Attention for Target Words ('dividend', 'dividends'): {avg_target_attention:.4f}")
# print(f"Normalized Average Attention for Comparison Words: {avg_comparison_attention:.4f}")
# print(f"Normalized Average Attention for Other Tokens: {avg_other_attention:.4f}")
#
# # Visualize the comparison
# labels = ['Dividend/Dividends', 'Comparison Words', 'Other Tokens']
# scores = [avg_target_attention, avg_comparison_attention, avg_other_attention]
#
# plt.bar(labels, scores, color=['blue', 'green', 'orange'])
# plt.ylabel('Normalized Average Attention')
# plt.title('Comparison of Attention Scores')
# plt.show()

# ______________________________________________

def get_dataset(idx):
    def clean_text(example, idx):
        if idx in (6,7,8):
            example['text'] = example['text'].replace('&#39;', "'")
        if idx in (6,7):
            # Remove non-English characters using regular expression
            example['text'] = re.sub(r'[^A-Za-z0-9\s.,!?\'\"-]', '', example['text'])
        return example

    if idx == 0:  # fiqa-sentiment-classification
        df = pd.read_csv('Data/fiqa.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.rename_column('sentence', 'text')
        dataset = dataset.filter(lambda example: example['type'] == 'headline')
        dataset = clean_dataset(dataset, idx)
    elif idx == 1:  # financial_phrasebank_75_agree
        df = pd.read_csv('Data/Sentences75Agree.csv')
        dataset = Dataset.from_pandas(df)
        # FPB = load_dataset("financial_phrasebank", 'sentences_75agree')['train']
        # dataset = FPB.rename_column('sentence', 'text')
    elif idx == 2:  # Stock-Market Sentiment Dataset
        df = pd.read_csv('Data/Stock-Market Sentiment Dataset.csv')
        df.rename(columns={'Text': 'text', 'Sentiment': 'label'}, inplace=True)
        dataset = Dataset.from_pandas(df)
        dataset = clean_dataset(dataset, idx)
    elif idx == 3:  # Aspect based Sentiment Analysis for Financial News
        df = pd.read_csv('Data/Processed_Financial_News.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 4:  # FinGPT/fingpt-sentiment-train
        df = pd.read_csv("/cs_storage/orkados/Data/FinGPT_cleaned_dataset.csv")
        df.rename(columns={'input': 'text', 'output': 'label'}, inplace=True)
        dataset = Dataset.from_pandas(df)

    # BACK-TRANSLATED DATASETS
    elif idx == 5:  # Back-Translated fiqa-sentiment-classification
        df = pd.read_csv('Data/back_translated/translated_dataset_0.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 6:  # Back-Translated financial_phrasebank_75_agree
        df = pd.read_csv('Data/back_translated/translated_dataset_1.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda example: clean_text(example, idx))
    elif idx == 7:  # Back-Translated Stock-Market Sentiment Dataset
        df = pd.read_csv('Data/back_translated/translated_dataset_2.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda example: clean_text(example, idx))
    elif idx == 8:  # Back-Translated Aspect based Sentiment Analysis for Financial News
        df = pd.read_csv('Data/back_translated/translated_dataset_3.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda example: clean_text(example, idx))

    # NEGATION EDITED DATASETS
    elif idx == 9:  # negation_fiqa-sentiment-classification
        df = pd.read_csv('Data/negation_dataset_0.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 10:  # negation_financial_phrasebank_75_agree
        df = pd.read_csv('Data/negation_dataset_1.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 11:  # negation_Stock-Market Sentiment Dataset
        df = pd.read_csv('Data/negation_dataset_2.csv')
        dataset = Dataset.from_pandas(df)
    else:  # idx == 12: negation_Aspect based Sentiment Analysis for Financial News
        df = pd.read_csv('Data/negation_dataset_3.csv')
        dataset = Dataset.from_pandas(df)

    return dataset

# for idx in range(1, NUM_DATASETS):
#     dataset = get_dataset(idx)
#     percentages = calculate_true_label_statistics(dataset)
