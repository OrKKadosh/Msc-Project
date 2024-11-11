# Evaluates the pt models & base model over the eval_dataset without FT.
import json
import os

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, \
    DataCollatorWithPadding, TrainingArguments, Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_model0 = {"tokenizer": "FacebookAI/roberta-base",
          "model": AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis').to(device),
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}#distilroberta-FT-financial-news-sentiment-analysis
base_model1 = {"tokenizer": "KernAI/stock-news-distilbert",
          "model": AutoModelForSequenceClassification.from_pretrained('KernAI/stock-news-distilbert', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('KernAI/stock-news-distilbert'),
          "name": "stock-news-distilbert"}#stock-news-distilbert
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
#     "model_for_PT": PeftModel.from_pretrained(
#         LlamaForCausalLM.from_pretrained(
#             "NousResearch/Llama-2-13b-hf", trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
#         ),
#         "FinGPT/fingpt-sentiment_llama2-13b_lora"
#     ),
#     "name": "FinGPT"
# } #FinGPT
base_model4 = {
    "tokenizer": "SALT-NLP/FLANG-ELECTRA",
    "model": AutoModelForSequenceClassification.from_pretrained("SALT-NLP/FLANG-ELECTRA", num_labels=3).to(device),
    "model_for_PT": AutoModelForMaskedLM.from_pretrained("SALT-NLP/FLANG-ELECTRA").to(device),
    "name": "FLANG-ELECTRA"
}#FLANG-ELECTRA
base_models = [base_model0, base_model1, base_model2, base_model4] #skipped training finGPT for now.
eval_dataset = load_dataset('TO_INSERT_TEST_DATASET')
NUM_DATASETS = 5

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1']
    }

def encode_labels(example, ds, model_name=None):
    """
    Encode labels as integers. Adjust label mapping for FinBERT where:
    "0": "positive", "1": "negative", "2": "neutral"
    For other models: "0": "negative", "1": "neutral", "2": "positive"
    """
    # Default label mapping (negative: 0, neutral: 1, positive: 2)
    label_dict_default = {'negative': 0, 'neutral': 1, 'positive': 2}
    # FinBERT-specific label mapping (positive: 0, negative: 1, neutral: 2)
    finbert_label_dict = {'positive': 0, 'negative': 1, 'neutral': 2}

    # Choose label dict based on model name
    if model_name == 'Finbert':  # Adjust labels for FinBERT
        label_dict = finbert_label_dict
    else:
        label_dict = label_dict_default

    # Encode labels based on dataset (ds) and model
    if ds == 0:  # fiqa-sentiment-classification
        if example['score'] <= -0.4:
            example['label'] = label_dict['negative']
        elif example['score'] <= 0.5:
            example['label'] = label_dict['neutral']
        else:
            example['label'] = label_dict['positive']
    elif ds == 1 :
        example['label'] = label_dict[example['label']]
    elif ds == 2 or ds == 6:  # Stock-Market Sentiment Dataset
        # Use stock market sentiment dataset labels (already numbers)
        label_dict1 = {1: label_dict['positive'], -1: label_dict['negative']}
        example['label'] = label_dict1[example['label']]
    # elif ds == 3 : This dataset is already mapped to support 0-negative, 1-neutral, 2-positive
    #     example['label'] = label_dict[example['label']]
    elif ds == 4:
        example['label'] = label_dict[example['output']]  # Map output to label
    return example

def tokenize_function(tokenizer, examples):
    output = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    return output


def fine_tuning_fixed():

    for model in base_models:

        model_name = model['name']
        base_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        pt_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        rd_pt_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=model['tokenizer'], return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'directory': base_directory,
            'model': model['model'],
            'tokenizer': model['tokenizer'],
            'data_collator': base_collator
        }
        pre_train_model = {
            "name": model_name,
            "type": "pt",
            "directory": pt_directory,
            "model": pt_model,
            "tokenizer": pt_tokenizer,
            "data_collator": pt_data_collator,
        }
        rd_pre_train_model = {
            "name": model_name,
            "type": "rd_pt",
            "directory": rd_pt_directory,
            "model": rd_pt_model,
            "tokenizer": rd_pt_tokenizer,
            "data_collator": rd_pt_data_collator,
        }
        base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]

        # Set up training arguments
        # training_args = TrainingArguments(
        #     output_dir="./train_checkpoints",
        #     learning_rate=2e-5,
        #     per_device_train_batch_size=8,
        #     num_train_epochs=NUM_TRAIN_EPOCH,
        #     weight_decay=0.01,
        #     save_strategy="epoch",
        #     save_steps=500,
        # )
        # Set up evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)

            encoded_eval_dataset = eval_dataset.map(lambda x: encode_labels(x, 8, model_name=inner_model['name'])) #todo: sent 8 as ds to pass encoding
            tokenized_eval_dataset = encoded_eval_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

            # Initialize the Trainer for the evaluation phase
            trainer = Trainer(
                model=inner_model['model'],
                args=evaluation_args,
                eval_dataset=tokenized_eval_dataset,
                tokenizer=inner_model['tokenizer'],
                data_collator=inner_model['data_collator'],
                compute_metrics=compute_metrics,
            )

            evaluation_results = trainer.evaluate()

            results_with_model = {
                "Type": inner_model['type'],
                "model_name": inner_model['name'],
                "results": evaluation_results,
                "eval_args" : evaluation_args,
            }

            model_type = inner_model['type']

            # results_file_name = f'{model_name}_{model_type}.txt'
            # results_dir = f"./Evaluation_results/no_FT/"
            # results_file_path = os.path.join(results_dir, results_file_name)

            results_file_name = f"{eval_dataset['name']}.txt"
            results_dir = f"./Evaluation_results/no_FT{now}/{model_name}/{model_type}/"
            os.makedirs(results_dir, exist_ok=True)
            results_file_path = os.path.join(results_dir, results_file_name)

            with open(results_file_path, "w") as file:
                file.write(json.dumps(results_with_model, indent=4))

            print(f"Evaluation results for the un-FT model: {model_name} saved to {results_file_name}")
