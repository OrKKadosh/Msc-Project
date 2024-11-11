# RUNS ON BOTH FILES: eval_all_agree & eval_75_consent

from datetime import datetime

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
    save_directory = f'Data/Error_Analysis/FT/{now}/{model_name}/{model_type}'
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

    # os.makedirs(f'Data/Error_Analysis/FT/{model_name}/{model_type}', exist_ok=True)
    misclassified_df = eval_df[eval_df['true_label'] != eval_df['predicted_label']]
    # misclassified_df.to_csv(f'Data/Error_Analysis/FT/{now}/{model_name}/{model_type}/misclassified_{eval_dataset_name}.csv', index=False)
    misclassified_df.to_csv(f'{save_directory}/misclassified_{eval_dataset_name}.csv', index=False)

    # ___________________________Big Errors___________________________
    misclassified_pos_to_neg = eval_df[(eval_df['true_label'] == 2 & eval_df['predicted_label'] == 0)]
    misclassified_pos_to_neg.to_csv(f'{save_directory}/misclassified_pos_to_neg_{eval_dataset_name}.csv', index=False)

    misclassified_neg_to_pos = eval_df[(eval_df['true_label'] == 0 & eval_df['predicted_label'] == 2)]
    misclassified_neg_to_pos.to_csv(f'{save_directory}/misclassified_neg_to_pos_{eval_dataset_name}.csv', index=False)

    big_errors_path =  f'{save_directory}/misclassified_big_errors_report_{eval_dataset_name}.txt'

    with open(big_errors_path, 'w') as file:
        file.write(f"Number of errors where the true_label is 2 and the predicted_label is 0 : {len(misclassified_pos_to_neg)} \n")
        file.write(f"Number of errors where the true_label is 0 and the predicted_label is 2 : {len(misclassified_neg_to_pos)}")


    # ___________________________Confusion Matrix___________________________
    cm = confusion_matrix(eval_df['true_label'], eval_df['predicted_label'], labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "neutral", "positive"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(f"{save_directory}/confusion_matrix_{eval_dataset_name}.png")  # Save as an image
    plt.show()  # Optionally display the plot


    # ___________________________Classification report___________________________
    report = classification_report(eval_df['true_label'], eval_df['predicted_label'],target_names=["negative", "neutral", "positive"])
    print(report)  # Display report in console
    with open(f"{save_directory}/classification_report_{eval_dataset_name}.txt","w") as file:
        file.write(report)


    # ___________________________FP & FN___________________________
    false_positives = eval_df[(eval_df['true_label'] != 0) & (eval_df['predicted_label'] == 0)]  # Model predicts negative, but should be neutral/positive
    false_positives.to_csv(f'{save_directory}/false_positives_{eval_dataset_name}.csv', index=False)

    false_negatives = eval_df[(eval_df['true_label'] == 0) & (eval_df['predicted_label'] != 0)]  # Model fails to predict negative
    false_negatives.to_csv(f'{save_directory}/false_negatives_{eval_dataset_name}.csv', index=False)


    # ___________________________Word cloud for misclassified texts___________________________
    misclassified_text = " ".join(misclassified_df['text'].values)
    wordcloud = WordCloud(width=800, height=400).generate(misclassified_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Save the plot as a file
    plt.savefig(f'{save_directory}/misclassified_wordcloud_{eval_dataset_name}.png', format="png", dpi=300)
    plt.show()


    # ___________________________Low confidence misclassification___________________________
    # Calculate the 25th percentile of the confidence scores
    low_confidence_threshold = np.percentile(eval_df['confidence'], 25)
    high_confidence_threshold = np.percentile(eval_df['confidence'], 75)

    low_confidence_misclassifications = eval_df[(eval_df['true_label'] != eval_df['predicted_label']) & (eval_df['confidence'] < low_confidence_threshold)]
    low_confidence_misclassifications['Low_confidence_threshold'] = low_confidence_threshold
    high_confidence_misclassifications = eval_df[(eval_df['true_label'] != eval_df['predicted_label']) & (eval_df['confidence'] >= high_confidence_threshold)]
    high_confidence_misclassifications['High_confidence_threshold'] = high_confidence_threshold

    low_confidence_misclassifications.to_csv(f'{save_directory}/low_confidence_misclassifications_{eval_dataset_name}.csv',index=False)
    high_confidence_misclassifications.to_csv(f'{save_directory}/high_confidence_misclassifications_{eval_dataset_name}.csv', index=False)

    avg_confidence = eval_df['confidence'].sum() / len(eval_df)
    with open(f'Data/Error_Analysis/FT/{now}/{model_name}/{model_type}/confidence_average_{eval_dataset_name}.txt', 'w') as file:
        file.write(f'The Average confidence score for this dataset : {avg_confidence}')


    # ___________________________True Neutral to Predicted Positive___________________________
    misclassified_neu_to_pos = eval_df[(eval_df['true_label'] == 1 & eval_df['predicted_label'] == 2)]
    misclassified_neu_to_pos.to_csv(f'Data/Error_Analysis/FT/{now}/{model_name}/{model_type}/misclassified_neu_to_pos.csv', index=False)

    # ___________________________Text Length___________________________
    length_save_directory = f'Data/Error_Analysis/FT/{now}/by_text_length/{model_name}/{model_type}'
    os.makedirs(length_save_directory, exist_ok=True)

    # Calculate the 33rd and 66th percentiles (quantiles) for text length
    short_threshold = eval_df['text_length'].quantile(0.33)
    medium_threshold = eval_df['text_length'].quantile(0.66)

    # Classify errors by text length categories based on calculated thresholds
    short_text_errors = eval_df[(eval_df['text_length'] <= short_threshold) & (eval_df['true_label'] != eval_df['predicted_label'])]
    medium_text_errors = eval_df[(eval_df['text_length'] > short_threshold) & (eval_df['text_length'] <= medium_threshold) & (eval_df['true_label'] != eval_df['predicted_label'])]
    long_text_errors = eval_df[(eval_df['text_length'] > medium_threshold) & (eval_df['true_label'] != eval_df['predicted_label'])]

    # # Classify errors by text length categories
    # short_text_errors = eval_df[(eval_df['text_length'] <= 61) & (eval_df['true_label'] != eval_df['predicted_label'])]
    # median_text_errors = eval_df[(eval_df['text_length'] <= 68) & (eval_df['text_length'] > 61) & (
    #             eval_df['true_label'] != eval_df['predicted_label'])]
    # long_text_errors = eval_df[(eval_df['text_length'] > 68) & (eval_df['true_label'] != eval_df['predicted_label'])]

    # Save the errors in separate files
    short_text_errors.to_csv(f'{length_save_directory}/short_length_errors_{eval_dataset_name}.csv', index=False)
    medium_text_errors.to_csv(f'{length_save_directory}/median_length_errors_{eval_dataset_name}.csv', index=False)
    long_text_errors.to_csv(f'{length_save_directory}/long_length_errors_{eval_dataset_name}.csv', index=False)

    # Calculate the total number of examples for each text length category
    short_text_total = eval_df[eval_df['text_length'] <= 61]
    median_text_total = eval_df[(eval_df['text_length'] <= 68) & (eval_df['text_length'] > 61)]
    long_text_total = eval_df[eval_df['text_length'] > 68]

    # Calculate error rates for each length category
    short_error_rate = len(short_text_errors) / len(short_text_total) * 100 if len(short_text_total) > 0 else 0
    median_error_rate = len(medium_text_errors) / len(median_text_total) * 100 if len(median_text_total) > 0 else 0
    long_error_rate = len(long_text_errors) / len(long_text_total) * 100 if len(long_text_total) > 0 else 0

    # Define the path for saving the error rate summary
    error_rate_summary_path = f'{length_save_directory}/error_rate_summary_{eval_dataset_name}.txt'

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


eval_consent_75_df = pd.read_csv('Data/test_datasets/split_eval_test/consent_75_eval.csv').apply(convert_labels_to_int, axis=1)
eval_consent_75 = Dataset.from_pandas(eval_consent_75_df)

eval_all_agree_df = pd.read_csv('Data/test_datasets/split_eval_test/all_agree_eval.csv').apply(convert_labels_to_int, axis=1)
eval_all_agree = Dataset.from_pandas(eval_all_agree_df)


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
    output = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
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

models_names = ['distilroberta', 'distilbert', 'finbert']
models_types = ['base', 'pt', 'rd_pt']
eval_datasets = [{'dataset': eval_all_agree, 'name': 'eval_all_agree'}, {'dataset': eval_consent_75, 'name': 'eval_consent_75'}]


def perform_error_analysis_all_agree(eval_dataset):

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
            model = AutoModelForSequenceClassification.from_pretrained(save_directory)
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
                break


for eval_dataset in eval_datasets:
    perform_error_analysis_all_agree(eval_dataset)

