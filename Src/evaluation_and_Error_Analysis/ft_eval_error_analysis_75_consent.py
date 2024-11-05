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

#     "0": "negative","1": "neutral","2": "positive"
base_model0 = {"tokenizer": "FacebookAI/roberta-base",
          "model": AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis').to(device),
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}#distilroberta-FT-financial-news-sentiment-analysis

#     "0": "negative","1": "neutral","2": "positive"
base_model1 = {"tokenizer": "KernAI/stock-news-distilbert",
          "model": AutoModelForSequenceClassification.from_pretrained('KernAI/stock-news-distilbert', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('KernAI/stock-news-distilbert'),
          "name": "stock-news-distilbert"}#stock-news-distilbert

# "0": "positive", "1": "negative", "2": "neutral"
base_model2 = {"tokenizer": "bert-base-uncased",
          "model": AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('ProsusAI/finbert').to(device),
          "name": "Finbert"}#FinBert
# base_model3 = {
#     "tokenizer": "NousResearch/Llama-2-13b-hf",
#     "model": PeftModel.from_pretrained(
#         LlamaForCausalLM.from_pretrained(
#             "NousResearch/Llama-2-13b-hf", trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
#         ),
#         "FinGPT/fingpt-sentiment_llama2-13b_lora"
#     ),
#     "name": "FinGPT"}#FinGPT

# base_model3 = {
#     "tokenizer": "NousResearch/Llama-2-13b-hf",
#     "model": PeftModel.from_pretrained(
#         LlamaForCausalLM.from_pretrained(
#             "NousResearch/Llama-2-13b-hf", trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
#         ),
#         "FinGPT/fingpt-sentiment_llama2-13b_lora"
#     ),
#     "model_for_PT": PeftModel.from_pretrained(
#         LlamaForCausalLM.from_pretrained(
#             "NousResearch/Llama-2-13b-hf", trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
#         ),
#         "FinGPT/fingpt-sentiment_llama2-13b_lora"
#     ),
#     "name": "FinGPT"
# } #FinGPT

# base_model4 = {
#     "tokenizer": "SALT-NLP/FLANG-ELECTRA",
#     "model": AutoModelForSequenceClassification.from_pretrained("SALT-NLP/FLANG-ELECTRA", num_labels=3).to(device),
#     "model_for_PT": AutoModelForMaskedLM.from_pretrained("SALT-NLP/FLANG-ELECTRA").to(device),
#     "name": "FLANG-ELECTRA"
# }#FLANG-ELECTRA
# base_models = [base_model0, base_model1, base_model2, base_model4]
base_models = [base_model0, base_model1, base_model2]
NUM_DATASETS = 5
NUM_TRAIN_EPOCH = 3

def error_analysis(model_name, model_type,data, eval_dataset,eval_dataset_name):
    print(f"Starts Error_Analysis on {model_name} of type: {model_type}")

    eval_df = pd.DataFrame({
        'text': eval_dataset['text'],
        'true_label': eval_dataset['label'],
        'predicted_label': data['predictions']
    })
    os.makedirs(f'Data/Error_Analysis/FT/{model_name}/{model_type}', exist_ok=True)
    misclassified_df = eval_df[eval_df['true_label'] != eval_df['predicted_label']]
    misclassified_df.to_csv(f'Data/Error_Analysis/FT/{model_name}/{model_type}/misclassified_{eval_dataset_name}.csv', index=False)

    # CONFUSION MATRIX
    cm = confusion_matrix(eval_df['true_label'], eval_df['predicted_label'], labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "neutral", "positive"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(f"Data/Error_Analysis/FT/{model_name}/{model_type}/confusion_matrix_{eval_dataset_name}.png")  # Save as an image
    plt.show()  # Optionally display the plot

    # classification report
    report = classification_report(eval_df['true_label'], eval_df['predicted_label'],target_names=["negative", "neutral", "positive"])
    with open(f"Data/Error_Analysis/FT/{model_name}/{model_type}/classification_report_{eval_dataset_name}.txt","w") as file:
        file.write(report)

    # FP & FN
    false_positives = eval_df[(eval_df['true_label'] != 0) & (eval_df['predicted_label'] == 0)]  # Model predicts negative, but should be neutral/positive
    false_positives.to_csv(f'Data/Error_Analysis/FT/{model_name}/{model_type}/false_positives_{eval_dataset_name}.csv', index=False)
    false_negatives = eval_df[(eval_df['true_label'] == 0) & (eval_df['predicted_label'] != 0)]  # Model fails to predict negative
    false_negatives.to_csv(f'Data/Error_Analysis/FT/{model_name}/{model_type}/false_negatives_{eval_dataset_name}.csv', index=False)

    # BY LENGTH
    eval_df['text_length'] = eval_df['text'].apply(lambda x: len(x.split()))
    short_text_errors = eval_df[(eval_df['text_length'] <= 61) & (eval_df['true_label'] != eval_df['predicted_label'])]
    median_text_errors = eval_df[(eval_df['text_length'] <= 68)& (eval_df['text_length'] > 61) & (eval_df['true_label'] != eval_df['predicted_label'])]
    long_text_errors = eval_df[(eval_df['text_length'] > 68) & (eval_df['true_label'] != eval_df['predicted_label'])]
    os.makedirs(f'Data/Error_Analysis/FT/by_text_length/{model_name}/{model_type}', exist_ok=True)
    short_text_errors.to_csv(f'Data/Error_Analysis/FT/by_text_length/{model_name}/{model_type}/short_length_{eval_dataset_name}.csv', index=False)
    median_text_errors.to_csv(f'Data/Error_Analysis/FT/by_text_length/{model_name}/{model_type}/median_length_{eval_dataset_name}.csv', index=False)
    long_text_errors.to_csv(f'Data/Error_Analysis/FT/by_text_length/{model_name}/{model_type}/long_length_{eval_dataset_name}.csv', index=False)

    # word cloud for misclassified texts
    misclassified_text = " ".join(misclassified_df['text'].values)
    wordcloud = WordCloud(width=800, height=400).generate(misclassified_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Save the plot as a file
    plt.savefig(f'Data/Error_Analysis/FT/{model_name}/{model_type}/misclassified_wordcloud_{eval_dataset_name}.png', format="png", dpi=300)
    plt.show()

    # LOW CONFIDENCE MISCLASSIFICATIONS
    probs = np.max(data['logits'], axis=1)  # Get the max probability for each prediction
    eval_df['confidence'] = probs
    low_confidence_misclassifications = eval_df[(eval_df['true_label'] != eval_df['predicted_label']) & (eval_df['confidence'] < 0.6)]
    low_confidence_misclassifications.to_csv(f'Data/Error_Analysis/FT/{model_name}/{model_type}/low_confidence_misclassifications_{eval_dataset_name}.csv',index=False)

def convert_labels_to_int(example):
    # Convert the labels to integers
    example['label'] = int(example['label'])
    return example


eval_consent_75_df = pd.read_csv('Data/test_datasets/split_eval_test/consent_75_eval.csv').apply(convert_labels_to_int, axis=1)
eval_consent_75 = Dataset.from_pandas(eval_consent_75_df)

eval_all_agree_df = pd.read_csv('Data/test_datasets/split_eval_test/all_agree_eval.csv').apply(convert_labels_to_int, axis=1)
eval_all_agree = Dataset.from_pandas(eval_all_agree_df)


predictions_dict = {
    'distilRoberta': [
        {'type': 'base', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'rd_pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}}
    ],
    'distilBert': [
        {'type': 'base', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'rd_pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}}
    ],
    'finBert': [
        {'type': 'base', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}},
        {'type': 'rd_pt', 'preds': {'predictions': [], 'labels': [], 'logits': []}}
    ],
    'Electra': [
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

    # Determine the main model name and access the correct dictionary based on model_type
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

    # Locate the right nested dictionary by model type and store predictions, labels, and logits
    for model in predictions_dict[model_key]:
        if model['type'] == model_type:
            model['preds']['predictions'] = predictions
            model['preds']['labels'] = labels
            model['preds']['logits'] = logits
            break

    # needed while using the FINBERT & base_stock-news-distilbert, since its labels are not matching
    if 'Finbert' in model_name:
        id2label = {0: 2, 1: 0, 2: 1}
        mapped_predictions = [id2label[pred] for pred in predictions]
    elif 'stock-news-distilbert' in model_name:
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

models_names = ['distilroberta-finetuned-financial-news-sentiment-analysis', 'stock-news-distilbert', 'Finbert']
models_types = ['base', 'pt', 'rd_pt']
eval_datasets = [{'dataset': eval_consent_75, 'name': 'eval_75_consent'}]


def perform_error_analysis_75_consent():

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

            idx = 0
            for eval_dataset in eval_datasets:

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
                    "eval_args": evaluation_args,
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/no_FT{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the un-FT model: {model_name} saved to {results_file_name}")
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
        model_name = get_model_name(result['model_name'])
        model_type = result['model_type']
        eval_dataset = result['eval_dataset']
        eval_dataset_name = result['eval_dataset_name']

        # Access the correct prediction data
        for type_data in predictions_dict[model_name]:
            if type_data['type'] == model_type:
                data = type_data['preds']
                error_analysis(model_name, model_type, data, eval_dataset, eval_dataset_name)
                break

    # __________________________
    # for model in base_models:
    #     model_name = model['name']
    #
    #     base_directory = f'./Saved_models/fine-tuned/{model_name}_base'
    #     pt_directory = f'./Saved_models/fine-tuned/{model_name}_pt'
    #     rd_pt_directory = f'./Saved_models/fine-tuned/{model_name}_rd_pt'
    #
    #     # Load fine-tuned models and tokenizers
    #     base_model = AutoModelForSequenceClassification.from_pretrained(base_directory)
    #     base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
    #
    #     pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory)
    #     pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
    #
    #     rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory)
    #     rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)
    #
    #     base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
    #     pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
    #     rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')
    #
    #     # Define models and settings
    #     base_and_pt_models = [
    #         {'name': model_name, 'type': 'base', 'model': base_model, 'tokenizer': base_tokenizer, 'data_collator': base_collator},
    #         {'name': model_name, 'type': 'pt', 'model': pt_model, 'tokenizer': pt_tokenizer, 'data_collator': pt_data_collator},
    #         {'name': model_name, 'type': 'rd_pt', 'model': rd_pt_model, 'tokenizer': rd_pt_tokenizer, 'data_collator': rd_pt_data_collator}
    #     ]
    #
    #     # Set up evaluation arguments
    #     evaluation_args = TrainingArguments(
    #         output_dir="./eval_checkpoints",
    #         per_device_eval_batch_size=2,
    #         logging_dir='./logs',
    #         do_eval=True,
    #         save_strategy="epoch",
    #     )
    #
    #     # Evaluate on each dataset
    #     for inner_model in base_and_pt_models:
    #         for eval_info in eval_datasets:
    #             eval_dataset = eval_info['dataset']
    #             eval_dataset_name = eval_info['name']
    #
    #             print(f"Evaluating {inner_model['name']} of type: {inner_model['type']} on dataset: {eval_dataset_name}")
    #
    #             # Tokenize the evaluation dataset
    #             tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)
    #
    #             # Set up the Trainer
    #             trainer = Trainer(
    #                 model=inner_model['model'],
    #                 args=evaluation_args,
    #                 eval_dataset=tokenized_eval_dataset,
    #                 tokenizer=inner_model['tokenizer'],
    #                 data_collator=inner_model['data_collator'],
    #                 compute_metrics=lambda eval_pred: compute_metrics_and_save_preds(eval_pred, model_name, inner_model['type']),
    #             )
    #
    #             # Perform the evaluation
    #             evaluation_results = trainer.evaluate()
    #
    #             # Save evaluation results and data for error analysis
    #             eval_results.append({
    #                 'model_name': model_name,
    #                 'model_type': inner_model['type'],
    #                 'eval_dataset': eval_dataset,
    #                 'eval_dataset_name': eval_dataset_name,
    #                 'evaluation_results': evaluation_results
    #             })
    #
    #             # Save results to file
    #             results_with_model = {
    #                 "Type": inner_model['type'],
    #                 "model_name": inner_model['name'],
    #                 "results": evaluation_results,
    #                 "eval_dataset": eval_dataset_name
    #             }
    #
    #             results_file_name = f'{model_name}_{inner_model["type"]}_{eval_dataset_name}.txt'
    #             results_dir = "./Evaluation_results/FT/"
    #             os.makedirs(results_dir, exist_ok=True)
    #             results_file_path = os.path.join(results_dir, results_file_name)
    #
    #             with open(results_file_path, "w") as file:
    #                 file.write(json.dumps(results_with_model, indent=4))
    #
    #             print(f"Evaluation results for the model: {model_name} of type: {inner_model['type']} on eval dataset: {eval_dataset_name} saved to {results_file_name}")
    #
    # # Run error analysis on the collected results
    # for result in eval_results:
    #     model_name = get_model_name(result['model_name'])
    #     model_type = result['model_type']
    #     eval_dataset = result['eval_dataset']
    #     eval_dataset_name = result['eval_dataset_name']
    #
    #     # Access the correct prediction data
    #     for type_data in predictions_dict[model_name]:
    #         if type_data['type'] == model_type:
    #             data = type_data['preds']
    #             error_analysis(model_name, model_type, data, eval_dataset, eval_dataset_name)
    #             break

perform_error_analysis_75_consent()
