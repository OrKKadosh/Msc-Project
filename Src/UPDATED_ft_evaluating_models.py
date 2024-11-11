from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import os
import re
from datetime import datetime

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
# base_models = [base_model0, base_model1, base_model2]
base_models = [base_model0]

# models_names = ['distilroberta-finetuned-financial-news-sentiment-analysis', 'stock-news-distilbert', 'Finbert']
models_names = ['electra']
# models_types = ['base', 'pt', 'rd_pt']
models_types = ['base']
NUM_DATASETS = 5
NUM_TRAIN_EPOCH = 3

def compute_manual_metrics(eval_preds):
    logits, labels = eval_preds
    # Convert logits to predictions (choosing the highest probability class)
    predictions = np.argmax(logits, axis=-1)

    # Convert labels to numpy array (if in torch tensor form)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Accuracy
    accuracy = accuracy_score(labels, predictions)

    # Precision, recall, F1 score (macro average)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')

    # Return the calculated metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def convert_labels_to_int(example):
    # Convert the labels to integers
    example['label'] = int(example['label'])
    return example

# all_agree_df = pd.read_csv('Data/test_datasets/all_agree_dataset.csv').apply(convert_labels_to_int, axis=1)
# all_agree_dataset = Dataset.from_pandas(all_agree_df)
#
# consent_75_df = pd.read_csv('Data/test_datasets/consent_75_dataset.csv').apply(convert_labels_to_int, axis=1)
# consent_75_dataset = Dataset.from_pandas(consent_75_df)

eval_consent_75_df = pd.read_csv('Data/test_datasets/split_eval_test/consent_75_eval.csv').apply(convert_labels_to_int, axis=1)
eval_consent_75 = Dataset.from_pandas(eval_consent_75_df)

eval_all_agree_df = pd.read_csv('Data/test_datasets/split_eval_test/all_agree_eval.csv').apply(convert_labels_to_int, axis=1)
eval_all_agree = Dataset.from_pandas(eval_all_agree_df)

# check_eval_df = pd.read_csv('Data/Processed_Financial_News.csv')
# check_eval_dataset = Dataset.from_pandas(check_eval_df)

eval_datasets = [eval_consent_75, eval_all_agree]


# def compute_metrics(eval_pred):
#     accuracy_metric = evaluate.load("accuracy")
#     precision_metric = evaluate.load("precision")
#     recall_metric = evaluate.load("recall")
#     f1_metric = evaluate.load("f1")
#
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
#     precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')
#     recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')
#     f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
#     return {
#         'accuracy': accuracy['accuracy'],
#         'precision': precision['precision'],
#         'recall': recall['recall'],
#         'f1': f1['f1']
#     }

def tokenize_function(tokenizer, examples):
    output = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    return output

accuracy_metric = evaluate.load("./local_metrics/accuracy")
precision_metric = evaluate.load("./local_metrics/precision")
recall_metric = evaluate.load("./local_metrics/recall")
f1_metric = evaluate.load("./local_metrics/f1")

def compute_metrics(eval_pred, model_name):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  # needed while using the FINBERT & base_stock-news-distilbert, since its labels are not matching
  if 'finbert' in model_name:
      id2label = {0: 2, 1: 0, 2: 1}
      mapped_predictions = [id2label[pred] for pred in predictions]
  elif 'distilbert' in model_name:
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

def eval_fine_tuned_models():

    for model_name in models_names:
        # Set up evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir=f"./eval_checkpoints/{model_name}",
            per_device_eval_batch_size=2,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )
        for model_type in models_types:

            # Load the Fine-Tuned model for evaluation
            save_directory = f'./Saved_models/fine-tuned/{model_name}_{model_type}'
            model = AutoModelForSequenceClassification.from_pretrained(save_directory).to(device)
            tokenizer = AutoTokenizer.from_pretrained(save_directory)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

            idx = 0
            for eval_dataset in eval_datasets:
                if (idx == 0):
                    eval_dataset_name = 'eval_75_consent'
                else:
                    eval_dataset_name = 'eval_all_agree'
                idx += 1

                print(f"Starts evaluating the FT model: {model_name} of type: {model_type} on dataset: {eval_dataset_name}")

                tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(tokenizer, x),batched=True)

                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=model,
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                evaluation_results = trainer.evaluate()

                results_with_model = {
                    "Type": model_type,
                    "model_name": model_name,
                    "results": evaluation_results,
                    "eval_dataset": eval_dataset_name,
                    "evaluation_args": evaluation_args.to_dict()
                }

                results_file_name = f'{eval_dataset_name}.txt'
                results_dir = f"./Evaluation_results/FT_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(
                    f"Evaluation results for the model: {model_name} of type: {model_type} on eval dataset: {eval_dataset_name} saved to {results_file_name}")


# fine-tuning each model(pt, rd_pt & base) on all ft datasets, saving the ft model, and evaluating the model on the evaluation dataset.
def evaluating():

    for model in base_models:

        model_name = model['name']

        base_directory = f'./Saved_models/fine-tuned/{model_name}_base'
        pt_directory = f'./Saved_models/fine-tuned/{model_name}_pt'
        rd_pt_directory = f'./Saved_models/fine-tuned/{model_name}_rd_pt'

        base_model = AutoModelForSequenceClassification.from_pretrained(base_directory)
        base_tokenizer = AutoTokenizer.from_pretrained(base_directory)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base_model,
            'tokenizer': base_tokenizer,
            'data_collator': base_collator
        }
        pre_train_model = {
            "name": model_name,
            "type": "pt",
            "model": pt_model,
            "tokenizer": pt_tokenizer,
            "data_collator": pt_data_collator,
        }
        rd_pre_train_model = {
            "name": model_name,
            "type": "rd_pt",
            "model": rd_pt_model,
            "tokenizer": rd_pt_tokenizer,
            "data_collator": rd_pt_data_collator,
        }
        base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]


        # Set up evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir=f"./eval_checkpoints/{model_name}/{model_type}",
            per_device_eval_batch_size=2,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )
        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            idx = 0
            for eval_dataset in eval_datasets:
                if (idx == 0):
                    eval_dataset_name = 'eval_75_consent'
                else:
                    eval_dataset_name = 'eval_all_agree'
                idx += 1

                print(f"Starts evaluating model: {inner_model['name']} of type: {inner_model['type']} of dataset: {eval_dataset_name}")
                tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)

                model_type = inner_model['type']
                save_directory = f'./Saved_models/fine-tuned/{model_name}_{model_type}'

                # Load the trained model for evaluation
                ft_model = AutoModelForSequenceClassification.from_pretrained(save_directory)

                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=ft_model,
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model['data_collator'],
                    compute_metrics=lambda eval_pred : compute_metrics(eval_pred, model_name),
                )

                evaluation_results = trainer.evaluate()

                results_with_model = {
                    "Type": inner_model['type'],
                    "model_name": inner_model['name'],
                    "results": evaluation_results,
                    "eval_dataset": eval_dataset_name,
                    "evaluation_args" : {
                                        "output_dir": "./eval_checkpoints",
                                        "per_device_eval_batch_size": 2,
                                        "logging_dir": './logs',
                                        "do_eval": True,
                                        "save_strategy": "epoch"
                                        }
                }

                results_file_name = f'{eval_dataset_name}.txt'
                results_dir = f"./Evaluation_results/FT_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} on eval dataset: {eval_dataset_name} saved to {results_file_name}")

eval_fine_tuned_models()