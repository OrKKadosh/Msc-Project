import json
import os
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, Dataset, load_dataset_builder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, \
    DataCollatorWithPadding, Trainer, TrainingArguments
import evaluate

print("running garbage.py")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

    # "0": "negative","1": "neutral","2": "positive"
base_model0 = {"tokenizer": "FacebookAI/roberta-base",
          "model": AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis').to(device),
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}#distilroberta-FT-financial-news-sentiment-analysis


base_models = [base_model0]
NUM_DATASETS = 5
NUM_TRAIN_EPOCH = 3
# eval_dataset = load_dataset('TO_INSERT_TEST_DATASET')

def convert_labels_to_int(example):
    # Convert the labels to integers
    example['label'] = int(example['label'])
    return example

accuracy_metric = evaluate.load("./local_metrics/accuracy")
precision_metric = evaluate.load("./local_metrics/precision")
recall_metric = evaluate.load("./local_metrics/recall")
f1_metric = evaluate.load("./local_metrics/f1")

def compute_metrics(eval_pred, model_name):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  # needed while using the FINBERT & base_stock-news-distilbert, since its labels are not matching
  if 'Finbert' in model_name:
      id2label = {0: 2, 1: 0, 2: 1}
      mapped_predictions = [id2label[pred] for pred in predictions]
  else: mapped_predictions = predictions


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


def get_dataset_try(idx):
    if idx == 0:  # fiqa-sentiment-classification
        # Manually load fiqa-sentiment data if available locally, otherwise raise an error
        train_path = '/tmp/cache/fiqa/fiqa_train.csv'
        valid_path = '/tmp/cache/fiqa/fiqa_valid.csv'
        test_path = '/tmp/cache/fiqa/fiqa_test.csv'

        if os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path):
            train_df = pd.read_csv(train_path).rename(columns={'sentence': 'text'})
            valid_df = pd.read_csv(valid_path).rename(columns={'sentence': 'text'})
            test_df = pd.read_csv(test_path).rename(columns={'sentence': 'text'})
            combined_df = pd.concat([train_df, valid_df, test_df])
            dataset = Dataset.from_pandas(combined_df)
            dataset = dataset.filter(lambda example: example['type'] == 'headline')
        else:
            raise FileNotFoundError("Please download fiqa dataset files and place them in '/tmp/cache/fiqa/'.")

    elif idx == 1:  # financial_phrasebank_75_agree
        # If caching issue arises, use a local CSV version; otherwise, load as usual
        try:
            dataset = load_dataset("financial_phrasebank", 'sentences_75agree')['train'].rename_column('sentence',
                                                                                                       'text')
        except NotImplementedError:
            phrasebank_path = '/tmp/cache/phrasebank/financial_phrasebank_75.csv'
            if os.path.exists(phrasebank_path):
                df = pd.read_csv(phrasebank_path).rename(columns={'sentence': 'text'})
                dataset = Dataset.from_pandas(df)
            else:
                raise FileNotFoundError(
                    "Please download the financial phrasebank 75-agree dataset and place it in '/tmp/cache/phrasebank/'.")

    elif idx == 2:  # Stock-Market Sentiment Dataset
        df = pd.read_csv('Data/Stock-Market Sentiment Dataset.csv')
        df.rename(columns={'Text': 'text', 'Sentiment': 'label'}, inplace=True)
        dataset = Dataset.from_pandas(df)

    # Add other datasets as needed without changing the existing loading method
    # ...

    return dataset
# fine-tuning each model(pt, rd_pt & base) on all ft datasets, saving the ft model, and evaluating the model on the evaluation dataset.
def fine_tuning_fixed():

    for model in base_models:

        model_name = model['name']
        pt_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        rd_pt_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

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

        base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]


        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./train_checkpoints",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            num_train_epochs=NUM_TRAIN_EPOCH,
            weight_decay=0.01,
            save_strategy="epoch",
            save_steps=500,
        )
        # Set up evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            do_eval=True,
            save_strategy="epoch",
        )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)

            print(f"Starting fine-tuning for model: {inner_model['name']} of type: {inner_model['type']}")
            # encoded_eval_dataset = eval_dataset.map(lambda x: encode_labels(x, idx, model_name=inner_model['name']))
            # tokenized_eval_dataset = encoded_eval_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

            for idx in range(NUM_DATASETS):  # Go through all datasets

                dataset = get_dataset(idx)  # Get dataset
                encoded_train_dataset = dataset.map(lambda x: encode_labels(x, idx, model_name=inner_model['name']))  # Encode train dataset

                tokenized_train_dataset = encoded_train_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)  # Tokenize train


                # Initialize the Trainer
                trainer = Trainer(
                    model=inner_model['model'],
                    args=training_args,
                    train_dataset=tokenized_train_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model["data_collator"],
                    compute_metrics=compute_metrics
                )
                print(f"Starting Fine-Tuning on dataset {idx} for model: {inner_model['name']} of type: {inner_model['type']}")

                # Train the model
                trainer.train()

            print(f"Fine-Tuning completed for model: {inner_model['name']} of type: {inner_model['type']}")

            model_type = inner_model['type']
            save_directory = f'./Saved_models/fine-tuned/{model_name}_{model_type}'
            trainer.save_model(save_directory)
            print(f"Model saved to {save_directory} after training on all datasets.")
            #
            # # Load the trained model for evaluation
            # ft_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
            #
            # # Initialize the Trainer for the evaluation phase
            # trainer = Trainer(
            #     model=ft_model,
            #     args=evaluation_args,
            #     eval_dataset=tokenized_eval_dataset,
            #     tokenizer=inner_model['tokenizer'],
            #     data_collator=inner_model['data_collator'],
            #     compute_metrics=compute_metrics,
            # )
            #
            # evaluation_results = trainer.evaluate()
            #
            # results_with_model = {
            #     "Type": inner_model['type'],
            #     "model_name": inner_model['name'],
            #     "results": evaluation_results
            # }
            #
            # results_file_name = f'{model_name}_{model_type}.txt'
            # results_dir = "./Evaluation_results/FT/"
            # results_file_path = os.path.join(results_dir, results_file_name)
            #
            # with open(results_file_path, "w") as file:
            #     file.write(json.dumps(results_with_model, indent=4))
            #
            # print(f"Evaluation results for the model: {model_name} saved to {results_file_name}")

def ft_base():
    model = AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device)
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    training_args = TrainingArguments(
        output_dir="./train_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=NUM_TRAIN_EPOCH,
        weight_decay=0.01,
        save_strategy="epoch",
        save_steps=500,
    )

    evaluation_args = TrainingArguments(
        output_dir="./eval_checkpoints",
        per_device_eval_batch_size=2,
        logging_dir='./logs',
        do_eval=True,
        save_strategy="epoch",
    )

    for idx in range(1):  # Go through all datasets

        dataset = get_dataset(idx)  # Get dataset
        encoded_train_dataset = dataset.map(
            lambda x: encode_labels(x, idx, "abc"))  # Encode train dataset

        tokenized_train_dataset = encoded_train_dataset.map(lambda x: tokenize_function(tokenizer, x),
                                                            batched=True)  # Tokenize train

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Train the model
        trainer.train()

    os.makedirs("./Saved_models/check/model0_base", exist_ok=True)
    save_directory = f'./Saved_models/check/model0_base'
    trainer.save_model(save_directory)
    print(f"fine-tuned model0 saved to {save_directory} after training on all datasets.")

    print("starts evaluating the ft model0")

    ft_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    eval_consent_75_df = pd.read_csv('Data/test_datasets/split_eval_test/consent_75_eval.csv').apply(convert_labels_to_int, axis=1)
    eval_dataset = Dataset.from_pandas(eval_consent_75_df)
    tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    # Initialize the Trainer for the evaluation phase
    trainer = Trainer(
        model=ft_model,
        args=evaluation_args,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, "abc"),
    )

    evaluation_results = trainer.evaluate()

    results_with_model = {
        "Type": 'base_ft',
        "model_name": 'distilroberta',
        "results": evaluation_results,
        "eval_dataset": '75_consent',
        "evaluation_args": {
            "output_dir": "./eval_checkpoints",
            "per_device_eval_batch_size": 2,
            "logging_dir": './logs',
            "do_eval": True,
            "save_strategy": "epoch"
        }
    }

    results_file_name = '75_consent.txt'
    results_dir = f"./Evaluation_results/check/'ft_roberta_base'/"
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, results_file_name)

    with open(results_file_path, "w") as file:
        file.write(json.dumps(results_with_model, indent=4))

    print(
        f"Evaluation results for the model: ft_roberta_base on eval dataset: '75_consent' saved to {results_file_name}")


#     __________________________________________
# Initialize the Trainer for the evaluation phase

    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'),
        args=evaluation_args,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, "abc"),
    )

    evaluation_results = trainer.evaluate()

    results_with_model = {
        "Type": 'base_no_ft',
        "model_name": 'distilroberta',
        "results": evaluation_results,
        "eval_dataset": '75_consent',
        "evaluation_args": {
            "output_dir": "./eval_checkpoints",
            "per_device_eval_batch_size": 2,
            "logging_dir": './logs',
            "do_eval": True,
            "save_strategy": "epoch"
        }
    }

    results_file_name = '75_consent.txt'
    results_dir = f"./Evaluation_results/check/'no_ft_roberta_base'/"
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, results_file_name)

    with open(results_file_path, "w") as file:
        file.write(json.dumps(results_with_model, indent=4))

    print(
        f"Evaluation results for the model: no_ft_roberta_base on eval dataset: '75_consent' saved to {results_file_name}")

# fine_tuning_fixed()
ft_base()
