import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, \
    DataCollatorWithPadding, Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification, AdamW
import evaluate
from peft import LoraConfig, TaskType, get_peft_model


print("Starts running Fine_tuning&Evaluating")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

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
    "model": AutoModelForSequenceClassification.from_pretrained("SALT-NLP/FLANG-ELECTRA", num_labels=3).to(device),
    "model_for_PT": AutoModelForMaskedLM.from_pretrained("SALT-NLP/FLANG-ELECTRA").to(device),
    "name": "electra"
}#FLANG-ELECTRA
base_models = [base_model0, base_model1, base_model2, base_model4]
NUM_DATASETS = 5
NUM_TRAIN_EPOCH = 3

def convert_labels_to_int(example):
    # Convert the labels to integers
    example['label'] = int(example['label'])
    return example


eval_consent_75_df = pd.read_csv('Data/test_datasets/split_eval_test/consent_75_eval.csv').apply(convert_labels_to_int, axis=1)
eval_dataset = Dataset.from_pandas(eval_consent_75_df)

# eval_all_agree_df = pd.read_csv('Data/test_datasets/split_eval_test/all_agree_eval.csv').apply(convert_labels_to_int, axis=1)
# eval_all_agree = Dataset.from_pandas(eval_all_agree_df)
# #
# eval_datasets = [eval_consent_75, eval_all_agree]



accuracy_metric = evaluate.load("./local_metrics/accuracy")
precision_metric = evaluate.load("./local_metrics/precision")
recall_metric = evaluate.load("./local_metrics/recall")
f1_metric = evaluate.load("./local_metrics/f1")

def compute_metrics(eval_pred, model_name):
    print("entered compute_metrics")
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

def encode_labels_2_labels(example, ds, model_name=None):
    """
    Encode labels for datasets, removing negative labels and keeping only neutral and positive for 2-label classification.
    """
    # print(example['label'])
    # Default label mapping (negative: 0, neutral: 1, positive: 2)
    label_dict = {'neutral': 0, 'positive': 1}
    # prev_label = example['score']
    # Encode labels based on dataset (ds) and model
    if ds == 0:  # fiqa-sentiment-classification
        if example['score'] <= -0.4:
            example['label'] = label_dict['negative']
        elif example['score'] <= 0.5:
            example['label'] = label_dict['neutral']
        else:
            example['label'] = label_dict['positive']
    elif ds == 1:
        example['label'] = label_dict[example['label']]
    elif ds == 3 : #This dataset is already mapped to support 0-negative, 1-neutral, 2-positive
        label_dict3 = {1: 0, 2: 1}
        example['label'] = label_dict3[example['label']]
    elif ds == 4:
        example['label'] = label_dict[example['label']]  # Map output to label
    # print(f"prev label: {prev_label}, updated label: {example['label']}")
    return example

# neutral - 0, positive - 1, negative - 2
def create_synth_dataset_with_dividend():
    from datasets import load_dataset
    import csv
    dataset = load_dataset("TimKoornstra/synthetic-financial-tweets-sentiment", split="train",
                           streaming=True).rename_column("tweet", "text").rename_column("sentiment", "label")
    with open("synth_dataset_with_dividend.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])

        for example in dataset:
            text = example['text']
            if "dividend" in text or "dividends" in text:
                writer.writerow([text, example['label']])


def create_neutral_synth_dataset():
    from datasets import load_dataset
    import csv
    dataset = load_dataset("TimKoornstra/synthetic-financial-tweets-sentiment", split="train",
                           streaming=True).rename_column("tweet", "text").rename_column("sentiment", "label")
    with open("synth_neutral_dataset.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])

        for example in dataset:
            if example["label"] == 0:
                writer.writerow([example["text"], example['label']])

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

def get_dataset_2_labels(idx):
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
        dataset = dataset.filter(lambda example: example['score'] > -0.4) #removes the negative samples

    elif idx == 1:  # financial_phrasebank_75_agree
        df = pd.read_csv('Data/Sentences75Agree.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.filter(lambda example: example['label'] != 0)  # removes the negative samples
        # FPB = load_dataset("financial_phrasebank", 'sentences_75agree')['train']
        # dataset = FPB.rename_column('sentence', 'text')
        dataset = dataset.filter(lambda example: example['label'] != 'negative') #removes the negative samples
    elif idx == 3:  # Aspect based Sentiment Analysis for Financial News
        df = pd.read_csv('Data/Processed_Financial_News.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.filter(lambda example: example['label'] != 0)  # removes the negative samples
    elif idx == 4:  # FinGPT/fingpt-sentiment-train
        df = pd.read_csv("/cs_storage/orkados/Data/FinGPT_cleaned_dataset.csv")
        df.rename(columns={'input': 'text', 'output': 'label'}, inplace=True)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.filter(lambda example: example['label'] != 'negative')  # removes the negative samples

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

def fine_tuning_fixed_2_labels():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./train_checkpoints/{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=NUM_TRAIN_EPOCH,
        weight_decay=0.01,
        save_strategy="epoch",
        save_steps=500,
    )

    print(f"Starting fine-tuning for model")

    for idx in range(NUM_DATASETS):  # Go through all datasets
        if idx == 2: continue #Stock-Market Sentiment Dataset has only positive or negative labels
        dataset = get_dataset_2_labels(idx)  # Get dataset
        # Encode labels
        encoded_train_dataset = dataset.map(
            lambda x: encode_labels_2_labels(x, idx, model_name=model_name)
        )

        # Tokenize train dataset
        tokenized_train_dataset = encoded_train_dataset.map(
            lambda x: tokenize_function(tokenizer, x), batched=True
        )
        # print("Tokenized dataset example:", tokenized_train_dataset[0])
        # print("Dataset columns:", tokenized_train_dataset.column_names)
        # print("Unique labels in tokenized_train_dataset:", set(tokenized_train_dataset['label']))
        # print("passed tokenization")

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            data_collator=data_collator,
            compute_metrics= compute_metrics,
        )
        # compute_metrics = lambda eval_pred: compute_metrics(eval_pred, model_name),

        print(f"Starting Fine-Tuning on dataset {idx} for model")

        # Train the model
        trainer.train()

    print(f"Fine-Tuning completed for model")

    model_type = 'base'
    os.makedirs(f"./Saved_models/ft&eval_2_labels/{model_name}/{model_type}", exist_ok=True)
    save_directory = f'./Saved_models/ft&eval_2_labels/{model_name}/{model_type}'
    trainer.save_model(save_directory)

    print(f"Model saved to {save_directory} after training on all datasets.")


import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        pt = (probs * targets_one_hot).sum(dim=-1)

        # Clamp the probabilities to avoid log(0)
        pt = torch.clamp(pt, min=1e-9)  # Ensures that pt is never smaller than 1e-9

        # Compute the focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels from inputs
        labels = inputs["labels"]

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate focal loss
        loss = self.focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


def fine_tuning_fixed_focal_loss():

    for model in base_models:

        model_name = model['name']
        pt_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        rd_pt_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
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
            output_dir=f"./train_checkpoints/{model_name}",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=NUM_TRAIN_EPOCH,
            weight_decay=0.01,
            save_strategy="epoch",
            save_steps=500,
        )
        # # Set up evaluation arguments
        # evaluation_args = TrainingArguments(
        #     output_dir="./eval_checkpoints",
        #     per_device_eval_batch_size=2,
        #     logging_dir='./logs',
        #     do_eval=True,
        #     save_strategy="epoch",
        # )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)

            print(f"Starting fine-tuning for model: {inner_model['name']} of type: {inner_model['type']}")
            tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

            # __________Using Lora for Electra model__________
            if "electra" in model_name:
                lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                                     lora_dropout=0.05)
                chosen_model = get_peft_model(inner_model["model"], lora_config)  # applying LORA
            else:
                chosen_model = inner_model["model"]

            for idx in range(NUM_DATASETS):  # Go through all datasets

                dataset = get_dataset(idx)  # Get dataset
                encoded_train_dataset = dataset.map(lambda x: encode_labels(x, idx, model_name=inner_model['name']))  # Encode train dataset

                tokenized_train_dataset = encoded_train_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)  # Tokenize train


                # Initialize the Trainer
                trainer = FocalLossTrainer(
                    model=chosen_model,
                    args=training_args,
                    train_dataset=tokenized_train_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model["data_collator"],
                    compute_metrics=lambda eval_pred : compute_metrics(eval_pred, model_name),
                )
                print(f"Starting Fine-Tuning on dataset {idx} for model: {inner_model['name']} of type: {inner_model['type']}")

                # Train the model
                trainer.train()

            print(f"Fine-Tuning completed for model: {inner_model['name']} of type: {inner_model['type']}")

            model_type = inner_model['type']
            os.makedirs(f"./Saved_models/ft&eval_focal_loss/{model_name}/{model_type}", exist_ok=True)
            save_directory = f'./Saved_models/ft&eval_focal_loss/{model_name}/{model_type}'
            trainer.save_model(save_directory)

            print(f"Model saved to {save_directory} after training on all datasets.")

            # print(f"Starts evaluating {model_name} of type: {model_type}")
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
            #     compute_metrics=lambda eval_pred : compute_metrics(eval_pred, model_name),
            # )
            #
            # evaluation_results = trainer.evaluate()
            #
            # results_with_model = {
            #     "Type": inner_model['type'],
            #     "model_name": inner_model['name'],
            #     "results": evaluation_results,
            #     "eval_dataset": '75_consent',
            #     "evaluation_args": {
            #         "output_dir": "./eval_checkpoints",
            #         "per_device_eval_batch_size": 2,
            #         "logging_dir": './logs',
            #         "do_eval": True,
            #         "save_strategy": "epoch"
            #     }
            # }
            #
            # results_file_name = '75_consent.txt'
            # results_dir = f"./Evaluation_results/fine-tuned/{model_name}_{model_type}/"
            # os.makedirs(results_dir, exist_ok=True)
            # results_file_path = os.path.join(results_dir, results_file_name)
            #
            # with open(results_file_path, "w") as file:
            #     file.write(json.dumps(results_with_model, indent=4))
            #
            # print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")


from torch.optim.lr_scheduler import LambdaLR


# def length_based_lr_scheduler(optimizer, base_lr, max_length):
#     """
#     Create a custom learning rate scheduler that adjusts the learning rate
#     based on input text lengths.
#     """
#
#     def lr_lambda(batch_avg_length):
#         # Scale down the learning rate for longer texts
#         return max(0.5, 1 - (batch_avg_length / max_length))
#
#     return LambdaLR(optimizer, lr_lambda)
#
#
# def filter_batch(batch):
#     return {
#         "input_ids": torch.tensor(batch["input_ids"]) if isinstance(batch["input_ids"], list) else batch["input_ids"],
#         "attention_mask": torch.tensor(batch["attention_mask"]) if isinstance(batch["attention_mask"], list) else batch["attention_mask"],
#         "token_type_ids": torch.tensor(batch.get("token_type_ids", None)) if "token_type_ids" in batch and isinstance(batch["token_type_ids"], list) else batch.get("token_type_ids", None)
#     }
#
# def fine_tuning_fixed_adjusting_learning_rate():
#
#     for model in base_models:
#         model_name = model['name']
#         pt_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
#
#         base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
#         base_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
#
#         base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
#
#         training_args = TrainingArguments(
#             output_dir=f"./train_checkpoints/{model_name}",
#             learning_rate=2e-5,  # Initial base learning rate
#             per_device_train_batch_size=8,
#             num_train_epochs=NUM_TRAIN_EPOCH,
#             weight_decay=0.01,
#             save_strategy="epoch",
#             save_steps=500,
#         )
#
#         print(f"Starting fine-tuning for model: {model_name}")
#
#         for idx in range(NUM_DATASETS):  # Iterate over datasets
#             dataset = get_dataset(idx)  # Load dataset
#             tokenized_train_dataset = dataset.map(lambda x: tokenize_function(base_tokenizer, x), batched=True)
#
#             # Initialize optimizer and custom scheduler
#             optimizer = AdamW(base_model.parameters(), lr=training_args.learning_rate)
#             scheduler = length_based_lr_scheduler(
#                 optimizer=optimizer,
#                 base_lr=training_args.learning_rate,
#                 max_length=512  # Adjust based on your tokenizer's max sequence length
#             )
#
#             def compute_batch_avg_length(batch):
#                 # Compute the average token length for a batch
#                 print(f"Batch: {batch}")
#                 print(f"input_ids: {batch['input_ids']}")
#                 return torch.tensor(len(batch['input_ids'])).float().mean()
#
#
#             # Fine-tuning loop
#             for batch in tokenized_train_dataset:
#                 avg_length = compute_batch_avg_length(batch)
#                 filtered_batch = filter_batch(batch)
#                 optimizer.zero_grad()
#                 outputs = base_model(**filtered_batch)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step(avg_length)  # Update learning rate based on avg length
#
#             print(f"Fine-tuning completed for dataset {idx} and model {model_name}")
#
#         # Save the model
#         save_directory = f'./Saved_models/ft&eval_adjusting_learning_rate/{model_name}/'
#         os.makedirs(save_directory, exist_ok=True)
#         base_model.save_pretrained(save_directory)
#         print(f"Model saved to {save_directory}")


import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
)
import os


# Define custom learning rate scheduler
def length_based_lr_scheduler(optimizer, max_length):
    """
    Create a custom learning rate scheduler that adjusts the learning rate
    based on input text lengths.
    """

    def lr_lambda(batch_avg_length):
        # Scale down the learning rate for longer texts
        return max(0.5, 1 - (batch_avg_length / max_length))

    return LambdaLR(optimizer, lr_lambda)


# Filter and convert batch to model input
def filter_batch(batch):
    return {
        "input_ids": torch.tensor(batch["input_ids"]).to(device) if isinstance(batch["input_ids"], list) else batch[
            "input_ids"].to(device),
        "attention_mask": torch.tensor(batch["attention_mask"]).to(device) if isinstance(batch["attention_mask"],
                                                                                         list) else batch[
            "attention_mask"].to(device),
        "token_type_ids": torch.tensor(batch.get("token_type_ids", None)).to(device) if "token_type_ids" in batch and
                                                                                        batch[
                                                                                            "token_type_ids"] is not None else None,
    }


def fine_tuning_fixed_adjusting_learning_rate():
    for model in base_models:
        model_name = model["name"]
        pt_directory = f"./Saved_models/pre_trained/Pre-Trained_{model_name}"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        base_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors="pt")

        training_args = TrainingArguments(
            output_dir=f"./train_checkpoints/{model_name}",
            per_device_train_batch_size=8,
            num_train_epochs=NUM_TRAIN_EPOCH + 3,  # Increase epochs for longer texts
            weight_decay=0.01,
            save_strategy="epoch",
            save_steps=500,
        )

        print(f"Starting fine-tuning for model: {model_name}")

        for idx in range(NUM_DATASETS):
            dataset = get_dataset(idx)  # Load dataset
            encoded_train_dataset = dataset.map(
                lambda x: encode_labels(x, idx, model_name=model_name))  # Encode train dataset
            tokenized_train_dataset = dataset.map(lambda x: tokenize_function(base_tokenizer, x), batched=True)

            # Initialize optimizer and scheduler
            optimizer = AdamW(base_model.parameters(), lr=training_args.learning_rate)
            scheduler = length_based_lr_scheduler(
                optimizer=optimizer,
                max_length=512  # Adjust based on tokenizer's max sequence length
            )

            def compute_batch_avg_length(batch):
                """
                Compute the average token length for a batch.
                """
                input_ids = batch["input_ids"]

                if isinstance(input_ids, torch.Tensor):
                    if input_ids.ndimension() == 2:  # Correct 2D input (batch_size, seq_len)
                        seq_lengths = input_ids.shape[1]  # Sequence length is the second dimension
                    elif input_ids.ndimension() == 1:  # Handle 1D input (single sequence)
                        seq_lengths = input_ids.shape[0]
                    else:  # Unexpected dimensions
                        raise ValueError(f"Unexpected tensor shape for input_ids: {input_ids.shape}")
                elif isinstance(input_ids, list):  # Handle list of lists
                    if all(isinstance(seq, list) for seq in input_ids):  # List of tokenized sequences
                        seq_lengths = torch.tensor([len(seq) for seq in input_ids])
                    else:
                        raise ValueError(f"Unexpected list format for input_ids: {input_ids}")
                else:
                    raise TypeError(f"Unsupported type for input_ids: {type(input_ids)}")

                # Compute the mean length
                return seq_lengths.float().mean() if isinstance(seq_lengths, torch.Tensor) else float(seq_lengths)

            # Fine-tuning loop
            for i, batch in enumerate(tokenized_train_dataset):
                filtered_batch = filter_batch(batch)

                # Compute average input length
                avg_length = compute_batch_avg_length(filtered_batch)

                optimizer.zero_grad()
                outputs = base_model(**filtered_batch)
                loss = outputs.loss

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
                scheduler.step(avg_length)  # Update learning rate based on avg length

                if i % 10 == 0:  # Log progress every 10 batches
                    print(f"Batch {i}: Loss = {loss.item():.4f}, Avg Length = {avg_length.item():.2f}")

            print(f"Fine-tuning completed for dataset {idx} and model {model_name}")

        # Save the fine-tuned model
        save_directory = f"./Saved_models/ft&eval_adjusting_learning_rate/{model_name}/"
        os.makedirs(save_directory, exist_ok=True)
        base_model.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy

def fine_tuning_fixed_adjusting_lr_epochs():

    for model in base_models:

        model_name = model["name"]
        pt_directory = f"./Saved_models/pre_trained/Pre-Trained_{model_name}"
        rd_pt_directory = f"./Saved_models/pre_trained/Pre-Trained+RD_{model_name}"

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors="pt")
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors="pt")
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors="pt")

        base_and_pt_models = [
            {
                "name": model_name,
                "type": "base",
                "model": model["model"],
                "tokenizer": base_tokenizer,
                "data_collator": base_collator,
            }
        ]

        for inner_model in base_and_pt_models:

            print(f"Starting fine-tuning for model: {inner_model['name']} of type: {inner_model['type']}")

            for idx in range(NUM_DATASETS):  # Iterate over datasets

                dataset = get_dataset(idx)  # Get dataset
                encoded_train_dataset = dataset.map(
                    lambda x: encode_labels(x, idx, model_name=inner_model["name"])
                )

                tokenized_train_dataset = encoded_train_dataset.map(
                    lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True
                )

                # Calculate average number of tokens
                total_tokens = sum(len(sample["input_ids"]) for sample in tokenized_train_dataset)
                avg_tokens = total_tokens / len(tokenized_train_dataset)
                print(f"Average number of tokens for dataset {idx}: {avg_tokens}")

                def determine_params(sample_length, avg_tokens):
                    """Adjust learning rate and epochs based on the sample length."""
                    if sample_length > avg_tokens:  # Longer samples
                        return 1e-5, NUM_TRAIN_EPOCH + 3
                    else:  # Shorter samples
                        return 2e-5, NUM_TRAIN_EPOCH

                # Tokenize dataset and assign appropriate params
                def tokenize_with_params(sample):
                    """Calculate parameters for individual samples."""
                    sample_length = len(sample["input_ids"])
                    lr, epochs = determine_params(sample_length, avg_tokens)
                    sample["learning_rate"] = lr
                    sample["epochs"] = epochs
                    return sample

                tokenized_train_dataset = tokenized_train_dataset.map(tokenize_with_params)

                # Aggregate parameters for the dataset (use max epochs and shared LR for simplicity)
                learning_rate = max(sample["learning_rate"] for sample in tokenized_train_dataset)
                num_epochs = max(sample["epochs"] for sample in tokenized_train_dataset)

                # Set up training arguments
                training_args = TrainingArguments(
                    output_dir=f"./train_checkpoints/{model_name}",
                    learning_rate=learning_rate,
                    per_device_train_batch_size=8,
                    num_train_epochs=num_epochs,
                    weight_decay=0.01,
                    save_strategy=IntervalStrategy.EPOCH,
                    save_steps=500,
                )

                # Initialize the Trainer
                trainer = Trainer(
                    model=inner_model["model"],
                    args=training_args,
                    train_dataset=tokenized_train_dataset,
                    tokenizer=inner_model["tokenizer"],
                    data_collator=inner_model["data_collator"],
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                print(
                    f"Starting Fine-Tuning on dataset {idx} for model: {inner_model['name']} "
                    f"of type: {inner_model['type']} with LR={learning_rate} and Epochs={num_epochs}"
                )

                # Train the model
                trainer.train()

            print(f"Fine-Tuning completed for model: {inner_model['name']} of type: {inner_model['type']}")

            # Save the model
            model_type = inner_model["type"]
            os.makedirs(f"./Saved_models/ft&eval_adust_lr/{model_name}/{model_type}", exist_ok=True)
            save_directory = f"./Saved_models/ft&eval_adust_lr/{model_name}/{model_type}"
            trainer.save_model(save_directory)

            print(f"Model saved to {save_directory} after training on all datasets.")



# fine-tuning each model(pt, rd_pt & base) on all ft datasets, saving the ft model, and evaluating the model on the evaluation dataset.
def fine_tuning_fixed_with_attention_penalty():
    for model in base_models:
        model_name = model['name']
        pt_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        rd_pt_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'

        # Load tokenizers and models
        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_model = model['model']
        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)

        # Collators
        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create models dictionary
        models_dict = [
            {
                "name": model_name,
                "type": "base",
                "model": base_model,
                "tokenizer": base_tokenizer,
                "data_collator": base_collator,
            },
            {
                "name": model_name,
                "type": "pt",
                "model": pt_model,
                "tokenizer": pt_tokenizer,
                "data_collator": pt_data_collator,
            },
            {
                "name": model_name,
                "type": "rd_pt",
                "model": rd_pt_model,
                "tokenizer": rd_pt_tokenizer,
                "data_collator": rd_pt_data_collator,
            },
        ]

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./train_checkpoints/{model_name}",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=NUM_TRAIN_EPOCH,
            weight_decay=0.01,
            save_strategy="epoch",
            save_steps=500,
        )

        for inner_model in models_dict:  # Iterate over all models
            print(f"Starting fine-tuning for model: {inner_model['name']} of type: {inner_model['type']}")

            # tokenized_eval_dataset = eval_dataset.map(
            #     lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True
            # )

            for idx in range(NUM_DATASETS):  # Iterate over datasets
                dataset = get_dataset(idx)
                encoded_train_dataset = dataset.map(lambda x: encode_labels(x, idx, model_name=inner_model['name']))  # Encode train dataset
                tokenized_train_dataset = encoded_train_dataset.map(
                    lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True
                )

                def rename_label(example):
                    example["labels"] = example.pop("label")  # Rename 'label' to 'labels'
                    return example

                tokenized_train_dataset = tokenized_train_dataset.map(rename_label)

                class CustomTrainer(Trainer):
                    def __init__(self, *args, tokenizer=None, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.tokenizer = tokenizer  # Ensure tokenizer is stored in the trainer
                    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                        """
                        Custom loss function to include attention penalty.
                        """
                        labels = inputs.pop("labels")
                        outputs = model(**inputs, output_attentions=True)

                        # Cross-entropy loss
                        ce_loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)

                        # Extract attention weightss
                        attention_weights = outputs.attentions  # (num_layers, batch_size, num_heads, seq_len, seq_len)
                        input_ids = inputs["input_ids"]

                        # Compute attention penalty for "dividend" and "dividends"
                        target_token_ids = [self.tokenizer("dividend", add_special_tokens=False)["input_ids"][0],
                                            self.tokenizer("dividends", add_special_tokens=False)["input_ids"][0]]
                        penalty = 0.0
                        for layer_attention in attention_weights:  # Iterate over layers
                            for head_attention in layer_attention:  # Iterate over heads
                                for token_id in target_token_ids:  # Iterate over target tokens
                                    target_positions = (input_ids == token_id).nonzero(as_tuple=True)[1]
                                    if len(target_positions) > 0:
                                        target_attention = head_attention[:, :, target_positions].sum()
                                        total_attention = head_attention.sum()
                                        penalty += (target_attention / total_attention - 1 / input_ids.size(1)) ** 2

                        # Combine cross-entropy loss and penalty
                        total_loss = ce_loss + 0.1 * penalty  # Adjust penalty weight as needed
                        return (total_loss, outputs) if return_outputs else total_loss
                # Custom Trainer with attention penalty
                trainer = CustomTrainer(
                    model=inner_model["model"],
                    args=training_args,
                    train_dataset=tokenized_train_dataset,
                    tokenizer=inner_model["tokenizer"],
                    data_collator=inner_model["data_collator"],
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                print(f"Starting Fine-Tuning on dataset {idx} for model: {inner_model['name']} of type: {inner_model['type']}")
                trainer.train()
            print(f"Fine-Tuning completed for model: {inner_model['name']} of type: {inner_model['type']}")

            # Save the fine-tuned model
            save_directory = f'./Saved_models/ft_with_attention_penalty/{model_name}/{inner_model["type"]}'
            os.makedirs(save_directory, exist_ok=True)
            trainer.save_model(save_directory)
            print(f"Model saved to {save_directory} after training.")

def fine_tuning_fixed_with_masking_dividend():
    def mask_specific_tokens(inputs, tokenizer, mask_prob=0.2):# Mask specific tokens ('dividend', 'dividends') in a random subset of inputs.
        input_ids = torch.tensor(inputs["input_ids"])
        mask_token_id = tokenizer.mask_token_id  # ID for [MASK] token (e.g., for BERT)

        # Get token IDs for 'dividend' and 'dividends'
        target_tokens = ["dividend", "dividends"]
        target_token_ids = [tokenizer(word, add_special_tokens=False)["input_ids"][0] for word in target_tokens]

        for i in range(input_ids.size(0)):  # Iterate over each example
            if random.random() < mask_prob:  # Mask with given probability
                for token_id in target_token_ids:
                    token_positions = (input_ids[i] == token_id).nonzero(as_tuple=True)[0]
                    for pos in token_positions:  # Mask all occurrences of the token
                        input_ids[i, pos] = mask_token_id  # Replace with [MASK]
        inputs["input_ids"] = input_ids.tolist()
        return inputs

    for model in base_models:
        model_name = model['name']
        pt_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        rd_pt_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'

        # Load tokenizers and models
        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_model = model['model']
        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)

        # Collators
        base_collator = DataCollatorWithPadding(tokenizer=base_tokenizer, return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')

        # Create models dictionary
        models_dict = [
            {
                "name": model_name,
                "type": "base",
                "model": base_model,
                "tokenizer": base_tokenizer,
                "data_collator": base_collator,
            },
            {
                "name": model_name,
                "type": "pt",
                "model": pt_model,
                "tokenizer": pt_tokenizer,
                "data_collator": pt_data_collator,
            },
            {
                "name": model_name,
                "type": "rd_pt",
                "model": rd_pt_model,
                "tokenizer": rd_pt_tokenizer,
                "data_collator": rd_pt_data_collator,
            },
        ]

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./train_checkpoints/{model_name}",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=NUM_TRAIN_EPOCH,
            weight_decay=0.01,
            save_strategy="epoch",
            save_steps=500,
        )

        for inner_model in models_dict:  # Iterate over all models
            print(f"Starting fine-tuning for model: {inner_model['name']} of type: {inner_model['type']}")

            # tokenized_eval_dataset = eval_dataset.map(
            #     lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True
            # )

            for idx in range(NUM_DATASETS):  # Iterate over datasets
                dataset = get_dataset(idx)
                encoded_train_dataset = dataset.map(lambda x: encode_labels(x, idx, model_name=inner_model['name']))  # Encode train dataset
                tokenized_train_dataset = encoded_train_dataset.map(
                    lambda x: mask_specific_tokens(tokenize_function(inner_model["tokenizer"], x), tokenizer=inner_model["tokenizer"], mask_prob=0.2), batched=True)


                # Custom Trainer with attention penalty
                trainer = Trainer(
                    model=inner_model["model"],
                    args=training_args,
                    train_dataset=tokenized_train_dataset,
                    tokenizer=inner_model["tokenizer"],
                    data_collator=inner_model["data_collator"],
                    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, model_name),
                )

                print(f"Starting Fine-Tuning on dataset {idx} for model: {inner_model['name']} of type: {inner_model['type']}")
                trainer.train()
            print(f"Fine-Tuning completed for model: {inner_model['name']} of type: {inner_model['type']}")

            # Save the fine-tuned model
            save_directory = f'./Saved_models/ft_with_masking_dividend/{model_name}/{inner_model["type"]}'
            os.makedirs(save_directory, exist_ok=True)
            trainer.save_model(save_directory)
            print(f"Model saved to {save_directory} after training.")


def fine_tuning_fixed():

    for model in base_models:

        model_name = model['name']
        pt_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        rd_pt_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'

        base_tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory, num_labels=3).to(device)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory, num_labels=3).to(device)
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
        base_and_pt_models = [base_model]

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"./train_checkpoints/{model_name}",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=NUM_TRAIN_EPOCH,
            weight_decay=0.01,
            save_strategy="epoch",
            save_steps=500,
        )
        # # Set up evaluation arguments
        # evaluation_args = TrainingArguments(
        #     output_dir="./eval_checkpoints",
        #     per_device_eval_batch_size=2,
        #     logging_dir='./logs',
        #     do_eval=True,
        #     save_strategy="epoch",
        # )

        for inner_model in base_and_pt_models:  # Iterate over models (base, pre-trained, RD pre-trained)

            print(f"Starting fine-tuning for model: {inner_model['name']} of type: {inner_model['type']}")
            tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x),batched=True)  # Tokenize eval

            # __________Using Lora for Electra model__________
            if "electra" in model_name:
                lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                                     lora_dropout=0.05)
                chosen_model = get_peft_model(inner_model["model"], lora_config)  # applying LORA
            else:
                chosen_model = inner_model["model"]

            for idx in range(NUM_DATASETS):  # Go through all datasets

                dataset = get_dataset(idx)  # Get dataset
                encoded_train_dataset = dataset.map(lambda x: encode_labels(x, idx, model_name=inner_model['name']))  # Encode train dataset

                tokenized_train_dataset = encoded_train_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)  # Tokenize train


                # Initialize the Trainer
                trainer = Trainer(
                    model=chosen_model,
                    args=training_args,
                    train_dataset=tokenized_train_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model["data_collator"],
                    compute_metrics=lambda eval_pred : compute_metrics(eval_pred, model_name),
                )
                print(f"Starting Fine-Tuning on dataset {idx} for model: {inner_model['name']} of type: {inner_model['type']}")

                # Train the model
                trainer.train()

            print(f"Fine-Tuning completed for model: {inner_model['name']} of type: {inner_model['type']}")

            model_type = inner_model['type']
            os.makedirs(f"./Saved_models/ft&eval/{model_name}/{model_type}", exist_ok=True)
            save_directory = f'./Saved_models/ft&eval/{model_name}/{model_type}'
            trainer.save_model(save_directory)

            print(f"Model saved to {save_directory} after training on all datasets.")

            # print(f"Starts evaluating {model_name} of type: {model_type}")
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
            #     compute_metrics=lambda eval_pred : compute_metrics(eval_pred, model_name),
            # )
            #
            # evaluation_results = trainer.evaluate()
            #
            # results_with_model = {
            #     "Type": inner_model['type'],
            #     "model_name": inner_model['name'],
            #     "results": evaluation_results,
            #     "eval_dataset": '75_consent',
            #     "evaluation_args": {
            #         "output_dir": "./eval_checkpoints",
            #         "per_device_eval_batch_size": 2,
            #         "logging_dir": './logs',
            #         "do_eval": True,
            #         "save_strategy": "epoch"
            #     }
            # }
            #
            # results_file_name = '75_consent.txt'
            # results_dir = f"./Evaluation_results/fine-tuned/{model_name}_{model_type}/"
            # os.makedirs(results_dir, exist_ok=True)
            # results_file_path = os.path.join(results_dir, results_file_name)
            #
            # with open(results_file_path, "w") as file:
            #     file.write(json.dumps(results_with_model, indent=4))
            #
            # print(f"Evaluation results for the model: {model_name} of type: {model_type} saved to {results_dir}")

# fine_tuning_fixed()
# fine_tuning_fixed_2_labels()

# fine_tuning_fixed_focal_loss()
fine_tuning_fixed_with_masking_dividend()
# fine_tuning_fixed_focal_loss()

