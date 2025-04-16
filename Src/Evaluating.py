import gc
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, \
    DataCollatorWithPadding, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import evaluate

now = datetime.now()
now = now.strftime("%Y-%m-%d %H:%M:%S")


print("Starts running Evaluating")

# device = torch.device("cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#     "0": "negative","1": "neutral","2": "positive"
base_model0 = {"tokenizer": "FacebookAI/roberta-base",
          "model": AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis').to(device),
          "name": "distilroberta"}#distilroberta-FT-financial-news-sentiment-analysis

#     "0": "negative","1": "neutral","2": "positive"
base_model1 = {"tokenizer": "KernAI/stock-news-distilbert",
          "model": AutoModelForSequenceClassification.from_pretrained('KernAI/stock-news-distilbert', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('KernAI/stock-news-distilbert'),
          "name": "distilbert"}#stock-news-distilbert

# "0": "positive", "1": "negative", "2": "neutral"
base_model2 = {"tokenizer": "bert-base-uncased",
          "model": AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('ProsusAI/finbert').to(device),
          "name": "finbert"}#FinBert
# base_model3 = {
#     "tokenizer": "NousResearch/Llama-2-13b-hf",
#     "model": PeftModel.from_pretrained(
#         LlamaForCausalLM.from_pretrained(
#             "NousResearch/Llama-2-13b-hf", trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
#         ),
#         "FinGPT/fingpt-sentiment_llama2-13b_lora"
#     ),
#     "name": "FinGPT"}#FinGPT
# base_model4 = {
#     "tokenizer": "SALT-NLP/FLANG-ELECTRA",
#     "model": AutoModelForSequenceClassification.from_pretrained("SALT-NLP/FLANG-ELECTRA", num_labels=3).to(device),
#     "name": "FLANG-ELECTRA"} #Flang-Electra

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
    "model": AutoModelForSequenceClassification.from_pretrained("SALT-NLP/FLANG-ELECTRA", num_labels=3),
    "model_for_PT": AutoModelForMaskedLM.from_pretrained("SALT-NLP/FLANG-ELECTRA"),
    "name": "electra"
}#FLANG-ELECTRA
base_models = [base_model0, base_model1, base_model2, base_model4]
# base_models = [base_model0, base_model1, base_model2]
NUM_DATASETS = 5
NUM_TRAIN_EPOCH = 3

def convert_labels_to_int(example):
    # Convert the labels to integers
    example['label'] = int(example['label'])
    return example


eval_consent_75_df = pd.read_csv('Data/test_datasets/split_eval_test/consent_75_eval.csv').apply(convert_labels_to_int, axis=1)
eval_consent_75 = Dataset.from_pandas(eval_consent_75_df)

# FPB = load_dataset("financial_phrasebank", 'sentences_75agree')['train']
# _, eval_dataset = FPB.train_test_split(test_size=0.3, seed=1694).values()

eval_all_agree_df = pd.read_csv('Data/test_datasets/split_eval_test/all_agree_eval.csv').apply(convert_labels_to_int, axis=1)
eval_all_agree = Dataset.from_pandas(eval_all_agree_df)

eval_dataset_dict = [{'dataset' : eval_consent_75, 'name': 'eval_consent_75'}, {'dataset' : eval_all_agree, 'name': 'eval_all_agree'}]



accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred, model_name):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  # needed while using the FINBERT & base_stock-news-distilbert, since its labels are not matching
  # if 'finbert' in model_name:
  #     id2label = {0: 2, 1: 0, 2: 1}
  #     mapped_predictions = [id2label[pred] for pred in predictions]
  # elif 'distilbert' in model_name:
  #     id2label = {0: 1, 1: 0, 2: 2}
  #     mapped_predictions = [id2label[pred] for pred in predictions]
  # else:
  #     mapped_predictions = predictions

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


def tokenize_function(tokenizer, examples):
    output = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    return output


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


def encode_labels_depends_on_model(example, ds, model_name=None):
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
        example['label'] = label_dict[example['label']]  # Map output to label
    return example


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
        df = pd.read_csv("Data/FinGPT_cleaned_dataset.csv")
        df.rename(columns={'input': 'text', 'output': 'label'}, inplace=True)
        dataset = Dataset.from_pandas(df)

    elif idx == 5:  # aug_negative_dataset.csv
        df = pd.read_csv("Data/aug_negative_dataset.csv")
        dataset = Dataset.from_pandas(df)
    else:  # aug_neutral_dataset.csv
        df = pd.read_csv("Data/aug_neutral_dataset.csv")
        dataset = Dataset.from_pandas(df)

    # # BACK-TRANSLATED DATASETS
    # elif idx == 5:  # Back-Translated fiqa-sentiment-classification
    #     df = pd.read_csv('Data/back_translated/translated_dataset_0.csv')
    #     dataset = Dataset.from_pandas(df)
    # elif idx == 6:  # Back-Translated financial_phrasebank_75_agree
    #     df = pd.read_csv('Data/back_translated/translated_dataset_1.csv')
    #     dataset = Dataset.from_pandas(df)
    #     dataset = dataset.map(lambda example: clean_text(example, idx))
    # elif idx == 7:  # Back-Translated Stock-Market Sentiment Dataset
    #     df = pd.read_csv('Data/back_translated/translated_dataset_2.csv')
    #     dataset = Dataset.from_pandas(df)
    #     dataset = dataset.map(lambda example: clean_text(example, idx))
    # elif idx == 8:  # Back-Translated Aspect based Sentiment Analysis for Financial News
    #     df = pd.read_csv('Data/back_translated/translated_dataset_3.csv')
    #     dataset = Dataset.from_pandas(df)
    #     dataset = dataset.map(lambda example: clean_text(example, idx))
    #
    # # NEGATION EDITED DATASETS
    # elif idx == 9:  # negation_fiqa-sentiment-classification
    #     df = pd.read_csv('Data/negation_dataset_0.csv')
    #     dataset = Dataset.from_pandas(df)
    # elif idx == 10:  # negation_financial_phrasebank_75_agree
    #     df = pd.read_csv('Data/negation_dataset_1.csv')
    #     dataset = Dataset.from_pandas(df)
    # elif idx == 11:  # negation_Stock-Market Sentiment Dataset
    #     df = pd.read_csv('Data/negation_dataset_2.csv')
    #     dataset = Dataset.from_pandas(df)
    # else:  # idx == 12: negation_Aspect based Sentiment Analysis for Financial News
    #     df = pd.read_csv('Data/negation_dataset_3.csv')
    #     dataset = Dataset.from_pandas(df)

    return dataset


# fine-tuning each model(pt, rd_pt & base) on all ft datasets, saving the ft model, and evaluating the model on the evaluation dataset.
def evaluating():

    for model in base_models:

        model_name = model['name']

        base_directory = f"./Saved_models/ft&eval/{model_name}/base"

        pt_directory = f"./Saved_models/ft&eval/{model_name}/pt"
        rd_pt_directory = f"./Saved_models/ft&eval/{model_name}/rd_pt"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        # electra_base_model = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        # electra_base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
        # electra_base_collator = DataCollatorWithPadding(tokenizer=electra_base_tokenizer, return_tensors='pt')


        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
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
        # base_and_pt_models = [base_model]



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_check_0101():

    for model in base_models:

        model_name = model['name']

        base_directory = f"./Saved_models/ft&eval/{model_name}/base"

        # pt_directory = f"./Saved_models/ft&eval/{model_name}/pt"
        # rd_pt_directory = f"./Saved_models/ft&eval/{model_name}/rd_pt"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        # pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        # pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)


        # rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        # rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        # pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        # rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
            'tokenizer': base_tokenizer,
            'data_collator': base_collator
        }
        # pre_train_model = {
        #     "name": model_name,
        #     "type": "pt",
        #     "model": pt_model,
        #     "tokenizer": pt_tokenizer,
        #     "data_collator": pt_data_collator,
        # }
        # rd_pre_train_model = {
        #     "name": model_name,
        #     "type": "rd_pt",
        #     "model": rd_pt_model,
        #     "tokenizer": rd_pt_tokenizer,
        #     "data_collator": rd_pt_data_collator,
        # }
        base_and_pt_models = [base_model]
        # base_and_pt_models = [base_model]



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_balanced_labels_distribution():

    for model in base_models:

        model_name = model['name']

        base_directory = f"./Saved_models/ft_with_balanced_labels_dist_2025-01-05_15-14-50/{model_name}/base/"

        pt_directory = f"./Saved_models/ft_with_balanced_labels_dist_2025-01-05_15-14-50/{model_name}/pt/"
        # pt_directory = './Saved_models/pre_trained_with_old_pt_distilroberta/Pre-Trained_distilroberta-finetuned-financial-news-sentiment-analysis/'
        # pt_directory = f'./Saved_models/pre_trained_with_old_pt/Pre-Trained_{model_name}'
        rd_pt_directory = f"./Saved_models/ft_with_balanced_labels_dist_2025-01-05_15-14-50/{model_name}/rd_pt/"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        # electra_base_model = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        # electra_base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
        # electra_base_collator = DataCollatorWithPadding(tokenizer=electra_base_tokenizer, return_tensors='pt')


        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
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
        # base_and_pt_models = [base_model]



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_balanced_labels_distribution_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")
def evaluating_synth_dividend():

    for model in base_models:

        model_name = model['name']

        base_directory = f"./Saved_models/ft&eval_neu_neg_synth_dividend_2025-01-05_17-25-43/{model_name}/base"

        pt_directory = f"./Saved_models/ft&eval_neu_neg_synth_dividend_2025-01-05_17-25-43/{model_name}/pt"
        # pt_directory = './Saved_models/pre_trained_with_old_pt_distilroberta/Pre-Trained_distilroberta-finetuned-financial-news-sentiment-analysis/'
        # pt_directory = f'./Saved_models/pre_trained_with_old_pt/Pre-Trained_{model_name}'
        rd_pt_directory = f"./Saved_models/ft&eval_neu_neg_synth_dividend_2025-01-05_17-25-43/{model_name}/rd_pt"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        # electra_base_model = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        # electra_base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
        # electra_base_collator = DataCollatorWithPadding(tokenizer=electra_base_tokenizer, return_tensors='pt')


        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
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
        # base_and_pt_models = [base_model]



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_neu_neg_synth_dividend_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_synth_dividend_detailed_results_check():
    for model in base_models:
        model_name = model['name']

        # Model directories
        base_directory = f"./Saved_models/ft&eval_neu_neg_synth_dividend_2025-01-05_17-25-43/{model_name}/base/"
        pt_directory = f"./Saved_models/ft&eval_neu_neg_synth_dividend_2025-01-05_17-25-43/{model_name}/pt/"
        rd_pt_directory = f"./Saved_models/ft&eval_neu_neg_synth_dividend_2025-01-05_17-25-43/{model_name}/rd_pt/"

        # Tokenizers and models
        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries
        base_model = {'name': model_name, 'type': 'base', 'model': base, 'tokenizer': base_tokenizer, 'data_collator': base_collator}
        pre_train_model = {'name': model_name, 'type': 'pt', 'model': pt_model, 'tokenizer': pt_tokenizer, 'data_collator': pt_data_collator}
        rd_pre_train_model = {'name': model_name, 'type': 'rd_pt', 'model': rd_pt_model, 'tokenizer': rd_pt_tokenizer, 'data_collator': rd_pt_data_collator}
        base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]

        # Evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(
                    lambda x: tokenize_function(inner_model["tokenizer"], x),
                    batched=True
                )

                model_type = inner_model['type']
                print(f"Starts evaluating {model_name} of type: {model_type}")

                # Initialize the Trainer for evaluation
                trainer = Trainer(
                    model=inner_model['model'],
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model['data_collator'],
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                # Get predictions
                predictions = trainer.predict(tokenized_eval_dataset)
                logits = predictions.predictions
                true_labels = predictions.label_ids

                # Map logits to predicted labels
                predicted_labels = torch.argmax(torch.tensor(logits), dim=-1).cpu().numpy()

                # Extract texts from the dataset
                text_data = [x['text'] for x in eval_dataset['dataset']]

                # Verify dataset lengths match
                if len(text_data) != len(true_labels) or len(text_data) != len(predicted_labels):
                    raise ValueError("Mismatch in dataset lengths: text, true_labels, and predicted_labels")

                # Create a Pandas DataFrame
                results_df = pd.DataFrame({
                    'text': text_data,
                    'true_label': true_labels,
                    'predicted_label': predicted_labels
                })

                # Save results to CSV
                results_dir = f"./Evaluation_results/eval_synth_dividend_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                csv_file_path = os.path.join(results_dir, f"{eval_dataset['name']}_predictions.csv")

                results_df.to_csv(csv_file_path, index=False)
                print(f"Prediction results saved to {csv_file_path}")

def evaluating_masking_dividend():

    for model in base_models:

        model_name = model['name']

        base_directory = f"./Saved_models/ft_with_masking_dividend_2025-01-07_12-13-44/{model_name}/base"

        pt_directory = f"./Saved_models/ft_with_masking_dividend_2025-01-07_12-13-44/{model_name}/pt"
        # pt_directory = './Saved_models/pre_trained_with_old_pt_distilroberta/Pre-Trained_distilroberta-finetuned-financial-news-sentiment-analysis/'
        # pt_directory = f'./Saved_models/pre_trained_with_old_pt/Pre-Trained_{model_name}'
        rd_pt_directory = f"./Saved_models/ft_with_masking_dividend_2025-01-07_12-13-44/{model_name}/rd_pt"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        # electra_base_model = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        # electra_base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
        # electra_base_collator = DataCollatorWithPadding(tokenizer=electra_base_tokenizer, return_tensors='pt')


        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
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
        # base_and_pt_models = [base_model]



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_masking_dividend_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_attention_penalty():

    for model in base_models:

        model_name = model['name']

        base_directory = f"./Saved_models/ft_with_attention_penalty_2025-01-07_12-09-35/{model_name}/base"

        pt_directory = f"./Saved_models/ft_with_attention_penalty_2025-01-07_12-09-35/{model_name}/pt"
        # pt_directory = './Saved_models/pre_trained_with_old_pt_distilroberta/Pre-Trained_distilroberta-finetuned-financial-news-sentiment-analysis/'
        # pt_directory = f'./Saved_models/pre_trained_with_old_pt/Pre-Trained_{model_name}'
        rd_pt_directory = f"./Saved_models/ft_with_attention_penalty_2025-01-07_12-09-35/{model_name}/rd_pt"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        # electra_base_model = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        # electra_base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
        # electra_base_collator = DataCollatorWithPadding(tokenizer=electra_base_tokenizer, return_tensors='pt')


        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
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
        # base_and_pt_models = [base_model]



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_attention_penalty_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_balanced_labels_distribution_detailed_results_check():
    for model in base_models:
        model_name = model['name']

        # Model directories
        base_directory = f"./Saved_models/ft_with_balanced_labels_dist_2024-12-29_14-40-25/{model_name}/base/"
        pt_directory = f"./Saved_models/ft_with_balanced_labels_dist_2024-12-29_14-40-25/{model_name}/pt/"
        rd_pt_directory = f"./Saved_models/ft_with_balanced_labels_dist_2024-12-29_14-40-25/{model_name}/rd_pt/"

        # Tokenizers and models
        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries
        base_model = {'name': model_name, 'type': 'base', 'model': base, 'tokenizer': base_tokenizer, 'data_collator': base_collator}
        pre_train_model = {'name': model_name, 'type': 'pt', 'model': pt_model, 'tokenizer': pt_tokenizer, 'data_collator': pt_data_collator}
        rd_pre_train_model = {'name': model_name, 'type': 'rd_pt', 'model': rd_pt_model, 'tokenizer': rd_pt_tokenizer, 'data_collator': rd_pt_data_collator}
        base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]

        # Evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(
                    lambda x: tokenize_function(inner_model["tokenizer"], x),
                    batched=True
                )

                model_type = inner_model['type']
                print(f"Starts evaluating {model_name} of type: {model_type}")

                # Initialize the Trainer for evaluation
                trainer = Trainer(
                    model=inner_model['model'],
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model['data_collator'],
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                # Get predictions
                predictions = trainer.predict(tokenized_eval_dataset)
                logits = predictions.predictions
                true_labels = predictions.label_ids

                # Map logits to predicted labels
                predicted_labels = torch.argmax(torch.tensor(logits), dim=-1).cpu().numpy()

                # Extract texts from the dataset
                text_data = [x['text'] for x in eval_dataset['dataset']]

                # Verify dataset lengths match
                if len(text_data) != len(true_labels) or len(text_data) != len(predicted_labels):
                    raise ValueError("Mismatch in dataset lengths: text, true_labels, and predicted_labels")

                # Create a Pandas DataFrame
                results_df = pd.DataFrame({
                    'text': text_data,
                    'true_label': true_labels,
                    'predicted_label': predicted_labels
                })

                # Save results to CSV
                results_dir = f"./Evaluation_results/eval_balanced_labels_distribution_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                csv_file_path = os.path.join(results_dir, f"{eval_dataset['name']}_predictions.csv")

                results_df.to_csv(csv_file_path, index=False)
                print(f"Prediction results saved to {csv_file_path}")


def evaluating_length_feature():

    for model in base_models:

        model_name = model['name']

        base_directory = f"./Saved_models/ft_text_length_feature/{model_name}/base"

        pt_directory = f"./Saved_models/ft_text_length_feature/{model_name}/pt"
        # pt_directory = './Saved_models/pre_trained_with_old_pt_distilroberta/Pre-Trained_distilroberta-finetuned-financial-news-sentiment-analysis/'
        # pt_directory = f'./Saved_models/pre_trained_with_old_pt/Pre-Trained_{model_name}'
        rd_pt_directory = f"./Saved_models/ft_text_length_feature/{model_name}/rd_pt"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        # electra_base_model = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        # electra_base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
        # electra_base_collator = DataCollatorWithPadding(tokenizer=electra_base_tokenizer, return_tensors='pt')


        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
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
        # base_and_pt_models = [base_model]



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_length_feature_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_length_feature_just_roberta_base():

    for model in base_models:

        model_name = model['name']

        base_directory = f"./Saved_models/ft_text_length_feature/{model_name}/base"

        # pt_directory = './Saved_models/pre_trained_with_old_pt_distilroberta/Pre-Trained_distilroberta-finetuned-financial-news-sentiment-analysis/'
        # pt_directory = f'./Saved_models/pre_trained_with_old_pt/Pre-Trained_{model_name}'

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        # pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        # pt_tokenizer = AutoTokenizer.fro_pretrained(pt_directory)

        # electra_base_model = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        # electra_base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
        # electra_base_collator = DataCollatorWithPadding(tokenizer=electra_base_tokenizer, return_tensors='pt')


        # rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        # rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        # pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        # rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
            'tokenizer': base_tokenizer,
            'data_collator': base_collator
        }
        # pre_train_model = {
        #     "name": model_name,
        #     "type": "pt",
        #     "model": pt_model,
        #     "tokenizer": pt_tokenizer,
        #     "data_collator": pt_data_collator,
        # }
        # rd_pre_train_model = {
        #     "name": model_name,
        #     "type": "rd_pt",
        #     "model": rd_pt_model,
        #     "tokenizer": rd_pt_tokenizer,
        #     "data_collator": rd_pt_data_collator,
        # }
        # base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]
        base_and_pt_models = [base_model]



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_length_feature_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_length_feature_detailed_results_check_just_roberta_base():
    for model in base_models:
        model_name = model['name']

        base_directory = f"./Saved_models/ft_text_length_feature/{model_name}/base"
        # pt_directory = f"./Saved_models/ft_text_length_feature_2025-01-07_19-24-11/{model_name}/pt"
        # rd_pt_directory = f"./Saved_models/ft_text_length_feature_2025-01-07_19-24-11/{model_name}/rd_pt"
        #
        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        # pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        # pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        #
        # rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        # rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)
        #
        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        # pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        # rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        base_model = {'name': model_name, 'type': 'base', 'model': base, 'tokenizer': base_tokenizer, 'data_collator': base_collator}
        # pre_train_model = {"name": model_name, "type": "pt", "model": pt_model, "tokenizer": pt_tokenizer, "data_collator": pt_data_collator}
        # rd_pre_train_model = {"name": model_name, "type": "rd_pt", "model": rd_pt_model, "tokenizer": rd_pt_tokenizer, "data_collator": rd_pt_data_collator}
        #
        base_and_pt_models = [base_model]

        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(
                    lambda x: tokenize_function(inner_model["tokenizer"], x),
                    batched=True
                )

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")

                trainer = Trainer(
                    model=inner_model['model'],
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model['data_collator'],
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                # Get predictions
                predictions = trainer.predict(tokenized_eval_dataset)
                preds = predictions.predictions.argmax(axis=-1)
                true_labels = predictions.label_ids

                # Extract text from dataset for better readability
                eval_texts = [x['text'] for x in eval_dataset['dataset']]

                # Create a DataFrame for results
                results_df = pd.DataFrame({
                    'Text': eval_texts,
                    'True_Label': true_labels,
                    'Predicted_Label': preds
                })

                # Save CSV
                results_dir = f"./Evaluation_results/eval_length_feature_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                csv_file_path = os.path.join(results_dir, f"{eval_dataset['name']}.csv")

                results_df.to_csv(csv_file_path, index=False)

                print(f"Evaluation results (with predictions and true labels) saved to {csv_file_path}")

import pandas as pd

def evaluating_length_feature_detailed_results_check():
    for model in base_models:
        model_name = model['name']

        base_directory = f"./Saved_models/ft_text_length_feature_2025-01-07_19-24-11/{model_name}/base"
        pt_directory = f"./Saved_models/ft_text_length_feature_2025-01-07_19-24-11/{model_name}/pt"
        rd_pt_directory = f"./Saved_models/ft_text_length_feature_2025-01-07_19-24-11/{model_name}/rd_pt"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        base_model = {'name': model_name, 'type': 'base', 'model': base, 'tokenizer': base_tokenizer, 'data_collator': base_collator}
        pre_train_model = {"name": model_name, "type": "pt", "model": pt_model, "tokenizer": pt_tokenizer, "data_collator": pt_data_collator}
        rd_pre_train_model = {"name": model_name, "type": "rd_pt", "model": rd_pt_model, "tokenizer": rd_pt_tokenizer, "data_collator": rd_pt_data_collator}

        base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]

        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(
                    lambda x: tokenize_function(inner_model["tokenizer"], x),
                    batched=True
                )

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")

                trainer = Trainer(
                    model=inner_model['model'],
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model['data_collator'],
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                # Get predictions
                predictions = trainer.predict(tokenized_eval_dataset)
                preds = predictions.predictions.argmax(axis=-1)
                true_labels = predictions.label_ids

                # Extract text from dataset for better readability
                eval_texts = [x['text'] for x in eval_dataset['dataset']]

                # Create a DataFrame for results
                results_df = pd.DataFrame({
                    'Text': eval_texts,
                    'True_Label': true_labels,
                    'Predicted_Label': preds
                })

                # Save CSV
                results_dir = f"./Evaluation_results/eval_length_feature_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                csv_file_path = os.path.join(results_dir, f"{eval_dataset['name']}.csv")

                results_df.to_csv(csv_file_path, index=False)

                print(f"Evaluation results (with predictions and true labels) saved to {csv_file_path}")


def evaluating_replace_dividend():
    def replace_dividend(eval_dataset):
        # Define the replacement functions for share, cut, and pay
        def replace_dividend_lambda(sample, replacement):
            sample["text"] = re.sub(r"\bdividends?\b", replacement, sample["text"], flags=re.IGNORECASE)
            return sample

        # Apply the replacement functions to create separate datasets
        updated_dataset_share = eval_dataset.map(lambda sample: replace_dividend_lambda(sample, "share"))
        updated_dataset_cut = eval_dataset.map(lambda sample: replace_dividend_lambda(sample, "cut"))
        updated_dataset_pay = eval_dataset.map(lambda sample: replace_dividend_lambda(sample, "pay"))

        return updated_dataset_share, updated_dataset_pay, updated_dataset_cut

    for model in base_models:
        model_name = model['name']
        base_directory = f"./Saved_models/fine-tuned/{model_name}_base"
        pt_directory = f"./Saved_models/fine-tuned/{model_name}_pt"
        rd_pt_directory = f"./Saved_models/fine-tuned/{model_name}_rd_pt"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        pt = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        rd_pt = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_collator  = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
            'tokenizer': base_tokenizer,
            'data_collator': base_collator
        }
        pre_train_model = {
            "name": model_name,
            "type": "pt",
            "model": pt,
            "tokenizer": pt_tokenizer,
            "data_collator": pt_collator,
        }
        rd_pre_train_model = {
            "name": model_name,
            "type": "rd_pt",
            "model": rd_pt,
            "tokenizer": rd_pt_tokenizer,
            "data_collator": rd_pt_collator,
        }

        base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]


        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models
            model_type = inner_model['type']
            for eval_dataset in eval_dataset_dict:
                # Generate modified datasets with the different replacements
                updated_dataset_share, updated_dataset_pay, updated_dataset_cut = replace_dividend(
                    eval_dataset["dataset"])

                # Evaluate each updated dataset
                for updated_dataset, replacement_name in zip(
                        [updated_dataset_share, updated_dataset_pay, updated_dataset_cut],
                        ["share", "pay", "cut"],
                ):
                    print(f"Starts evaluating {model_name} with replacement: {replacement_name}")

                    # Tokenize the updated dataset
                    tokenized_eval_dataset = updated_dataset.map(
                        lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True
                    )

                    # Initialize the Trainer
                    trainer = Trainer(
                        model=inner_model['model'],
                        args=evaluation_args,
                        eval_dataset=tokenized_eval_dataset,
                        tokenizer=inner_model['tokenizer'],
                        data_collator=inner_model['data_collator'],
                        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                    )

                    evaluation_results = trainer.evaluate()

                    # Save evaluation results
                    results_with_model = {
                        "Type": inner_model['type'],
                        "model_name": inner_model['name'],
                        "replacement": replacement_name,
                        "results": evaluation_results,
                        "eval_dataset": eval_dataset['name'],
                        "evaluation_args": evaluation_args.to_dict(),
                    }

                    results_file_name = f"{eval_dataset['name']}_{replacement_name}.txt"
                    results_dir = f"./Evaluation_results/eval_replace_dividend_{now}/{model_name}/{model_type}/{replacement_name}/"
                    os.makedirs(results_dir, exist_ok=True)
                    results_file_path = os.path.join(results_dir, results_file_name)

                    with open(results_file_path, "w") as file:
                        file.write(json.dumps(results_with_model, indent=4))

                    print(
                        f"Evaluation results for {model_name} with replacement: {replacement_name} saved to {results_dir}")


def evaluating_using_focal_loss_ft():

    for model in base_models:

        model_name = model['name']

        # base_directory = f"./Saved_models/fine-tuned/{model_name}_base"
        base_directory = f"./Saved_models/ft&eval_focal_loss/{model_name}/base/"

        pt_directory = f"./Saved_models/ft&eval_focal_loss/{model_name}/pt/"
        # pt_directory = './Saved_models/pre_trained_with_old_pt_distilroberta/Pre-Trained_distilroberta-finetuned-financial-news-sentiment-analysis/'
        # pt_directory = f'./Saved_models/pre_trained_with_old_pt/Pre-Trained_{model_name}'
        rd_pt_directory = f"./Saved_models/ft&eval_focal_loss/{model_name}/rd_pt/"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        # electra_base_model = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        # electra_base_tokenizer = AutoTokenizer.from_pretrained(base_directory)
        # electra_base_collator = DataCollatorWithPadding(tokenizer=electra_base_tokenizer, return_tensors='pt')


        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': base,
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



        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_focal_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_using_sec_model():
    # Initialize the secondary model and tokenizer only ONCE
    secondary_model_path = "/cs_storage/orkados/Saved_models/ft&eval_2_labels/bert-base-uncased/base/"
    secondary_model = AutoModelForSequenceClassification.from_pretrained(
        secondary_model_path,
        num_labels=2,  # For Neutral vs Positive
        ignore_mismatched_sizes=True
    ).to(device)
    secondary_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def map_labels(model_name, predictions):
        """
        Map predictions based on the model's specific label encoding.
        """
        if 'finbert' in model_name:
            id2label = {0: 2, 1: 0, 2: 1}
            return [id2label[pred] for pred in predictions]
        elif 'distilbert' in model_name:
            id2label = {0: 1, 1: 0, 2: 2}
            return [id2label[pred] for pred in predictions]
        else:
            return predictions  # No mapping required

    def refine_prediction(primary_prediction, input_text):
        """
        Refines predictions using the secondary classifier for Neutral vs Positive.
        """
        if primary_prediction == 2:  # Positive prediction in primary model
            inputs = secondary_tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length").to(device)
            secondary_logits = secondary_model(**inputs).logits
            refined_prediction = torch.argmax(secondary_logits, dim=-1).item()

            # Map back to the original label space
            if refined_prediction == 0:  # Neutral in the secondary model
                return 1  # Neutral in the primary model
            elif refined_prediction == 1:  # Positive in the secondary model
                return 2  # Positive in the primary model

        return primary_prediction  # Return unchanged if not refined

    for model in base_models:
        model_name = model['name']

        base_directory = f"./Saved_models/fine-tuned/{model_name}_base"
        pt_directory = f"./Saved_models/fine-tuned/{model_name}_pt"
        rd_pt_directory = f"./Saved_models/fine-tuned/{model_name}_rd_pt"

        # Load tokenizers and models
        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        base = AutoModelForSequenceClassification.from_pretrained(base_directory, num_labels=3).to(device)
        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        # Data collators
        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Models for evaluation
        base_and_pt_models = [
            {"name": model_name, "type": "base", "model": base, "tokenizer": base_tokenizer, "data_collator": base_collator},
            {"name": model_name, "type": "pt", "model": pt_model, "tokenizer": pt_tokenizer, "data_collator": pt_data_collator},
            {"name": model_name, "type": "rd_pt", "model": rd_pt_model, "tokenizer": rd_pt_tokenizer, "data_collator": rd_pt_data_collator},
        ]

        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)

                model_type = inner_model['type']
                print(f"Starts evaluating {model_name} of type: {model_type}")

                # Initialize the Trainer for evaluation
                trainer = Trainer(
                    model=inner_model['model'],
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model['data_collator'],
                )

                # Run evaluation to get raw predictions
                predictions, labels, _ = trainer.predict(tokenized_eval_dataset)
                predictions = np.argmax(predictions, axis=1)

                # Map primary predictions to the correct label space
                mapped_predictions = map_labels(model_name, predictions)

                total_changes = 0
                true_label_matches = 0
                refined_predictions = []

                # Refine predictions where necessary
                for i, example in enumerate(eval_dataset['dataset']):
                    input_text = example['text']
                    primary_prediction = mapped_predictions[i]
                    refined_prediction = refine_prediction(primary_prediction, input_text)

                    if primary_prediction != refined_prediction:
                        total_changes += 1
                        if refined_prediction == int(labels[i]):  # Compare refined with original labels
                            true_label_matches += 1

                    refined_predictions.append(refined_prediction)

                # Convert all data to Python-native types for JSON serialization
                refined_predictions = [int(pred) for pred in refined_predictions]
                labels = [int(label) for label in labels]

                # Metrics for refined predictions
                refined_metrics = {
                    "accuracy": accuracy_metric.compute(predictions=refined_predictions, references=labels)["accuracy"],
                    "precision": precision_metric.compute(predictions=refined_predictions, references=labels, average='macro')["precision"],
                    "recall": recall_metric.compute(predictions=refined_predictions, references=labels, average='macro')["recall"],
                    "f1": f1_metric.compute(predictions=refined_predictions, references=labels, average='macro')["f1"],
                }

                primary_metrics = {
                    "accuracy": accuracy_metric.compute(predictions=mapped_predictions, references=labels)["accuracy"],
                    "precision": precision_metric.compute(predictions=mapped_predictions, references=labels, average='macro')["precision"],
                    "recall": recall_metric.compute(predictions=mapped_predictions, references=labels, average='macro')["recall"],
                    "f1": f1_metric.compute(predictions=mapped_predictions, references=labels, average='macro')["f1"],
                }

                print("Primary Metrics:", primary_metrics)
                print("Refined Metrics:", refined_metrics)

                # Save metrics and refined results
                results_file_name = f"{eval_dataset['name']}_refined_results.json"
                results_dir = f"./Evaluation_results/eval_with_sec_model_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)

                results_data = {
                    "primary_metrics": primary_metrics,
                    "refined_metrics": refined_metrics,
                    "summary": {
                        "total_changes": int(total_changes),  # Ensure serialization-friendly format
                        "true_label_matches": int(true_label_matches),
                    },
                    "details": [
                        {
                            "text": example["text"],
                            "true_label": int(labels[i]),  # Use original labels here
                            "primary_prediction": int(mapped_predictions[i]),
                            "refined_prediction": refined_predictions[i],
                        }
                        for i, example in enumerate(eval_dataset['dataset'])
                    ],
                }

                with open(os.path.join(results_dir, results_file_name), "w") as file:
                    json.dump(results_data, file, indent=4)

                print(f"Refined Evaluation results for {model_name} of type: {model_type} saved to {results_dir}")

#
# for eval_dataset in eval_datasets:
#     evaluating(eval_dataset)

def evaluating_pt_models_11_11():

    for model in base_models:

        model_name = model['name']

        # base_directory = f"./Saved_models/ft&eval/{model_name}/base"
        # pt_directory = f"./Saved_models/ft&eval/{model_name}/pt"
        pt_directory = f"./Saved_models/pre_trained_with_old_pt_{model_name}/Pre-Trained_{model_name}/"
        # pt_directory = f'./Saved_models/pre_trained_with_old_pt/Pre-Trained_{model_name}'
        rd_pt_directory = f"./Saved_models/pre_trained/Pre-Trained+RD_{model_name}/"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        # _____________________________________>__________________________________________
        classification_head = model['model'].classifier  # The classification head layer

        # Load the MLM model as a sequence classification model
        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)

        # Replace the classification head with the extracted head
        pt_model.classifier = classification_head
        # _____________________________________<__________________________________________
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        base_tokenizer = AutoTokenizer.from_pretrained(model['tokenizer'])

        # _____________________________________>__________________________________________
        # Load the MLM model as a sequence classification model
        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)

        # Replace the classification head with the extracted head
        rd_pt_model.classifier = classification_head
        # _____________________________________<__________________________________________

        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': model['model'],
            'tokenizer': base_tokenizer,
            'data_collator': base_collator
        } # only for ELECTRA
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
        # base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]
        base_and_pt_models = [base_model, pre_train_model]



        # Set up evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

                model_type = inner_model['type']

                print(f"Starts evaluating {model_name} of type: {model_type}")


                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=inner_model['model'],
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
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }


                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

def evaluating_pt_models_12_11():
    for model in base_models:
        model_name = model['name']
        pt_directory = f"./Saved_models/pre_trained_with_old_pt_{model_name}/Pre-Trained_{model_name}/"
        rd_pt_directory = f"./Saved_models/pre_trained/Pre-Trained+RD_{model_name}/"

        # Load base tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        classification_head = model['model'].classifier  # Extract classification head from base model

        # Load pre-trained models with sequence classification head
        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)

        # Attempt to load the classification head weights into pt_model and rd_pt_model
        try:
            pt_model.classifier.load_state_dict(classification_head.state_dict())
            print(f"Loaded classification head for {model_name} into pt_model successfully.")
        except RuntimeError as e:
            print(f"Could not load classification head for {model_name} into pt_model. Error: {e}")
            print("Using randomly initialized classifier head for pt_model.")

        try:
            rd_pt_model.classifier.load_state_dict(classification_head.state_dict())
            print(f"Loaded classification head for {model_name} into rd_pt_model successfully.")
        except RuntimeError as e:
            print(f"Could not load classification head for {model_name} into rd_pt_model. Error: {e}")
            print("Using randomly initialized classifier head for rd_pt_model.")

        # Tokenizers for the pre-trained models
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        # Data collators for padding
        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Model dictionaries for base, pre-trained, and RD pre-trained models
        base_model = {
            'name': model_name,
            'type': 'base',
            'model': model['model'],
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

        # Choose models for evaluation
        base_and_pt_models = [base_model, pre_train_model]

        # Set up evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)

                model_type = inner_model['type']
                print(f"Starts evaluating {model_name} of type: {model_type}")

                # Initialize Trainer for evaluation
                trainer = Trainer(
                    model=inner_model['model'],
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model['data_collator'],
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                evaluation_results = trainer.evaluate()
                results_with_model = {
                    "Type": inner_model['type'],
                    "model_name": inner_model['name'],
                    "results": evaluation_results,
                    "eval_dataset": eval_dataset['name'],
                    "evaluation_args": evaluation_args.to_dict()
                }

                results_file_name = f"{eval_dataset['name']}.txt"
                results_dir = f"./Evaluation_results/eval_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for {model_name} of type: {model_type} saved to {results_dir}")

count = 0
def evaluating_pt_models_24_11():
    # Initialize the secondary model and tokenizer only ONCE
    secondary_model_name = 'ProsusAI/finbert'
    secondary_model = AutoModelForSequenceClassification.from_pretrained(
        secondary_model_name,
        num_labels=2,  # For Neutral vs Positive
        ignore_mismatched_sizes=True
    ).to(device)
    secondary_tokenizer = AutoTokenizer.from_pretrained(secondary_model_name)

    def refine_prediction(primary_prediction, input_text):
        """
        Refines predictions using the secondary classifier for Neutral vs Positive.
        """
        if primary_prediction == 2:  # Positive prediction
            inputs = secondary_tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length").to(
                device)
            secondary_logits = secondary_model(**inputs).logits
            refined_prediction = torch.argmax(secondary_logits, dim=-1).item()

            # Map back to the original label space
            if refined_prediction == 0:  # Neutral in the secondary model
                return 1  # Neutral in the primary model
            elif refined_prediction == 1:  # Positive in the secondary model
                return 2  # Positive in the primary model

        return primary_prediction

    for model in base_models:
        model_name = model['name']
        pt_directory = f"./Saved_models/pre_trained_with_old_pt_{model_name}/Pre-Trained_{model_name}/"
        rd_pt_directory = f"./Saved_models/pre_trained/Pre-Trained+RD_{model_name}/"

        # Load base tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        # Load pre-trained models with sequence classification head
        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)

        # Tokenizers for the pre-trained models
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        # Data collators for padding
        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Model dictionaries for evaluation
        base_and_pt_models = [
            {"type": "base", "model": model['model'], "tokenizer": base_tokenizer, "data_collator": base_collator},
            {"type": "pt", "model": pt_model, "tokenizer": pt_tokenizer, "data_collator": pt_data_collator},
        ]

        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:
            for eval_dataset in eval_dataset_dict:
                tokenized_eval_dataset = eval_dataset['dataset'].map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)

                model_type = inner_model['type']
                print(f"Starts evaluating {model_name} of type: {model_type}")

                # Initialize Trainer for evaluation
                trainer = Trainer(
                    model=inner_model['model'],
                    args=evaluation_args,
                    eval_dataset=tokenized_eval_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model['data_collator'],
                )

                # Run evaluation to get raw predictions
                predictions, labels, _ = trainer.predict(tokenized_eval_dataset)
                predictions = np.argmax(predictions, axis=1)

                total_changes = 0
                true_label_matches = 0
                refined_predictions = []

                # Refine predictions where necessary
                for i, example in enumerate(eval_dataset['dataset']):
                    input_text = example['text']
                    primary_prediction = predictions[i]
                    refined_prediction = refine_prediction(primary_prediction, input_text)

                    if primary_prediction != refined_prediction:
                        total_changes += 1
                        if refined_prediction == int(example['label']):
                            true_label_matches += 1

                    refined_predictions.append(refined_prediction)
                # _______________________________DEBUG
                for i in range(len(refined_predictions)):
                    print(f"Index: {i},Primary Prediction: {predictions[i]}, Refined Prediction: {refined_predictions[i]}, True Label: {labels[i]}")
                # _______________________________DEBUG
                # Convert all data to Python-native types for JSON serialization
                refined_predictions = [int(pred) for pred in refined_predictions]
                labels = [int(label) for label in labels]

                # Metrics for refined predictions
                metrics = {
                    "accuracy": accuracy_metric.compute(predictions=refined_predictions, references=labels)["accuracy"],
                    "precision": precision_metric.compute(predictions=refined_predictions, references=labels, average='macro')["precision"],
                    "recall": recall_metric.compute(predictions=refined_predictions, references=labels, average='macro')["recall"],
                    "f1": f1_metric.compute(predictions=refined_predictions, references=labels, average='macro')["f1"],
                }

                primary_metrics = {
                    "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
                    "precision": precision_metric.compute(predictions=predictions, references=labels, average='macro')[
                        "precision"],
                    "recall": recall_metric.compute(predictions=predictions, references=labels, average='macro')[
                        "recall"],
                    "f1": f1_metric.compute(predictions=predictions, references=labels, average='macro')["f1"],
                }

                print("Primary Metrics:", primary_metrics)
                print("Refined Metrics:", metrics)

                # Save metrics and refined results
                results_file_name = f"{eval_dataset['name']}_refined_results.json"
                results_dir = f"./Evaluation_results/eval_{now}/{model_name}/{model_type}/"
                os.makedirs(results_dir, exist_ok=True)

                results_data = {
                    "metrics": metrics,
                    "summary": {
                        "count": count,
                        "total_changes": int(total_changes),  # Ensure serialization-friendly format
                        "true_label_matches": int(true_label_matches),
                    },
                    "details": [
                        {
                            "text": example["text"],
                            "true_label": int(example["label"]),
                            "primary_prediction": int(predictions[i]),
                            "refined_prediction": refined_predictions[i],
                        }
                        for i, example in enumerate(eval_dataset['dataset'])
                    ],
                }

                with open(os.path.join(results_dir, results_file_name), "w") as file:
                    json.dump(results_data, file, indent=4)

                print(f"Refined Evaluation results for {model_name} of type: {model_type} saved to {results_dir}")



# check that the mlm accuracy for the py model makes sense
def check_mlm_models():

    # Set up Trainer with evaluation arguments
    training_args = TrainingArguments(
        output_dir="./eval_checkpoints",
        per_device_eval_batch_size=4,
        logging_dir='./logs',
        do_eval=True,
        evaluation_strategy="epoch",
    )
    models_names = ['distilroberta', 'distilbert', 'finbert']

    pretrain_dataset = load_dataset('Lettria/financial-articles').remove_columns(['source_name', 'url', 'origin'])['train']
    pretrain_dataset = pretrain_dataset.rename_column("content", "text")
    split_dataset = pretrain_dataset.train_test_split(0.1, seed=1694)
    small_pt_dataset = split_dataset["test"]

    for model_name in models_names:
        # Load pre-trained MLM model and tokenizer
        pt_directory  = f"./Saved_models/pre_trained/Pre-Trained_{model_name}"
        pt_model = AutoModelForMaskedLM.from_pretrained(pt_directory)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        pt_tokenized_eval_dataset = small_pt_dataset.map(lambda x: tokenize_function(pt_tokenizer, x),batched=True)

        # rd_pt_directory = f"./Saved_models/pre_trained/Pre-Trained+RD_{model_name}"
        # rd_pt_model = AutoModelForMaskedLM.from_pretrained(rd_pt_directory)
        # rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)
        # rd_pt_tokenized_eval_dataset = small_pt_dataset.map(lambda x: tokenize_function(rd_pt_tokenizer, x),batched=True)


        # Mask tokens with DataCollatorForLanguageModeling
        pt_data_collator = DataCollatorForLanguageModeling(tokenizer=pt_tokenizer, mlm=True, mlm_probability=0.15)
        # rd_pt_data_collator = DataCollatorForLanguageModeling(tokenizer=rd_pt_tokenizer, mlm=True, mlm_probability=0.15)



        # Define evaluation function to calculate MLM accuracy
        def compute_mlm_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            # Only consider masked tokens for accuracy calculation
            mask = labels != -100
            correct_predictions = (predictions[mask] == labels[mask]).sum()
            total_predictions = mask.sum()

            accuracy = correct_predictions / total_predictions
            return {"mlm_accuracy": accuracy}

        # Initialize Trainer
        pt_trainer = Trainer(
            model=pt_model,
            args=training_args,
            eval_dataset=pt_tokenized_eval_dataset,
            data_collator=pt_data_collator,
            compute_metrics=compute_mlm_metrics,
        )

        # rd_pt_trainer = Trainer(
        #     model=rd_pt_model,
        #     args=training_args,
        #     eval_dataset=rd_pt_tokenized_eval_dataset,
        #     data_collator=rd_pt_data_collator,
        #     compute_metrics=compute_mlm_metrics,
        # )

        # Run evaluation and print results
        pt_eval_results = pt_trainer.evaluate()
        print("MLM Evaluation Results:", pt_eval_results)
        pt_perplexity = np.exp(pt_eval_results["eval_loss"])
        print("Perplexity:", pt_perplexity)

        # rd_pt_eval_results = rd_pt_trainer.evaluate()
        # print("MLM Evaluation Results:", rd_pt_eval_results)
        # rd_pt_perplexity = np.exp(rd_pt_eval_results["eval_loss"])
        # print("Perplexity:", rd_pt_perplexity)

def check_mlm_models_new():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Set up Trainer with evaluation arguments
    training_args = TrainingArguments(
        output_dir="./eval_checkpoints",
        per_device_eval_batch_size=2,  # Reduced batch size to fit memory
        logging_dir='./logs',
        do_eval=True,
        evaluation_strategy="epoch",
        fp16=True,  # Enable mixed-precision training to save memory
        dataloader_pin_memory=False
    )
    models_names = ['distilroberta', 'distilbert', 'finbert']

    # Load and prepare dataset
    pretrain_dataset = load_dataset('Lettria/financial-articles').remove_columns(['source_name', 'url', 'origin'])[
        'train']
    pretrain_dataset = pretrain_dataset.rename_column("content", "text")
    split_dataset = pretrain_dataset.train_test_split(0.1, seed=1694)
    small_pt_dataset = split_dataset["test"]

    for model_name in models_names:
        # Clear CUDA cache to avoid leftover allocations
        torch.cuda.empty_cache()

        # Load pre-trained MLM model and tokenizer
        pt_directory = f"./Saved_models/pre_trained/Pre-Trained_{model_name}"
        pt_model = AutoModelForMaskedLM.from_pretrained(pt_directory).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        pt_tokenized_eval_dataset = small_pt_dataset.map(lambda x: tokenize_function(pt_tokenizer, x), batched=True)

        # Mask tokens with DataCollatorForLanguageModeling
        pt_data_collator = DataCollatorForLanguageModeling(tokenizer=pt_tokenizer, mlm=True, mlm_probability=0.15)

        # Define evaluation function to calculate MLM accuracy
        def compute_mlm_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            # Only consider masked tokens for accuracy calculation
            mask = labels != -100
            correct_predictions = (predictions[mask] == labels[mask]).sum()
            total_predictions = mask.sum()

            accuracy = correct_predictions / total_predictions
            return {"mlm_accuracy": accuracy}

        # Initialize Trainer
        pt_trainer = Trainer(
            model=pt_model,
            args=training_args,
            eval_dataset=pt_tokenized_eval_dataset,
            data_collator=pt_data_collator,
            compute_metrics=compute_mlm_metrics,
        )

        # Evaluate model with no gradients to reduce memory usage
        with torch.no_grad():
            pt_eval_results = pt_trainer.evaluate()

        print("MLM Evaluation Results:", pt_eval_results)

        # Calculate perplexity
        pt_perplexity = np.exp(pt_eval_results["eval_loss"])
        print("Perplexity:", pt_perplexity)


def find_large_tensors(threshold=100 * 1024 ** 2):  # default threshold is 100 MB
    large_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                size = obj.numel() * obj.element_size()  # bytes
                if size >= threshold:
                    large_tensors.append((type(obj), size, obj.size()))
        except Exception as e:
            pass  # ignore any non-tensor objects
    return large_tensors

# evaluating_pt_models()
# evaluating_pt_models_12_11()
# evaluating_using_focal_loss_ft()
# evaluating_using_sec_model()
# evaluating_replace_dividend()
# evaluating()
# evaluating_balanced_labels_distribution()
# evaluating_length_feature_just_roberta_base()
# evaluating_length_feature_detailed_results_check_just_roberta_base()
# evaluating_synth_dividend_detailed_results_check()
# evaluating_check_0101()
# evaluating_masking_dividend()
evaluating_attention_penalty()
