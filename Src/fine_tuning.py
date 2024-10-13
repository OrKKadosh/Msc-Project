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
    "name": "FLANG-ELECTRA"
}#FLANG-ELECTRA
base_models = [base_model0, base_model1, base_model2, base_model4]
NUM_DATASETS = 5
NUM_TRAIN_EPOCH = 3
eval_dataset = load_dataset('TO_INSERT_TEST_DATASET')

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

def encode_labels(example, ds):

    # Encode labels as integers according to: 0-negative, 1-neutral, 2-positive

    label_dict2 = {1:2, -1:0}  # Stock-Market Sentiment Dataset
    if ds == 0:  #fiqa-sentiment-classification
        if example['score'] <= -0.4:
            example['label'] = 0
        elif example['score'] <= 0.5:
            example['label'] = 1
        else:
            example['label'] = 2
    elif ds == 2 or ds == 6 :  #Stock-Market Sentiment Dataset
        example['label'] = label_dict2[example['label']]

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
        train_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='train').rename_column('sentence', 'text')
        valid_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='valid').rename_column('sentence', 'text')
        test_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='test').rename_column('sentence', 'text')
        concatenate_dataset = concatenate_datasets([train_dataset, valid_dataset, test_dataset])
        dataset = concatenate_dataset.filter(lambda example: example['type'] == 'headline')
        dataset = clean_dataset(dataset, idx)
    elif idx == 1:  # financial_phrasebank_75_agree
        FPB = load_dataset("financial_phrasebank", 'sentences_75agree')['train']
        dataset = FPB.rename_column('sentence', 'text')
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

def fine_tuning():
    for inner_model in base_models:

        model_name = inner_model['name']
        base_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        pt_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        rd_pt_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'

        pt_model = AutoModelForSequenceClassification.from_pretrained(pt_directory)
        pt_tokenizer = AutoTokenizer.from_pretrained(pt_directory)

        rd_pt_model = AutoModelForSequenceClassification.from_pretrained(rd_pt_directory)
        rd_pt_tokenizer = AutoTokenizer.from_pretrained(rd_pt_directory)

        base_collator = DataCollatorWithPadding(tokenizer=inner_model['tokenizer'], return_tensors='pt')
        pt_data_collator = DataCollatorWithPadding(tokenizer=pt_tokenizer, return_tensors='pt')
        rd_pt_data_collator = DataCollatorWithPadding(tokenizer=rd_pt_tokenizer, return_tensors='pt')


        base_model = {
            'name' : model_name,
            'type' : 'base',
            'directory' : base_directory,
            'model' : inner_model['model'],
            'tokenizer' : inner_model['tokenizer'],
            'data_collator' : base_collator
        }
        pre_train_model = {
            "name" : model_name,
            "type" : "pt",
            "directory" : pt_directory,
            "model" : pt_model,
            "tokenizer" : pt_tokenizer,
            "data_collator" : pt_data_collator,
        }
        rd_pre_train_model = {
            "name" : model_name,
            "type" : "rd_pt",
            "directory" : rd_pt_directory,
            "model" : rd_pt_model,
            "tokenizer" : rd_pt_tokenizer,
            "data_collator" : rd_pt_data_collator,
        }
        base_and_pt_models = [base_model, pre_train_model, rd_pre_train_model]

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./train_checkpoints",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
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

        for idx in range(NUM_DATASETS): #includes the Back-Translated DS
            dataset = get_dataset(idx)
            encoded_train_dataset = dataset.map(lambda x: encode_labels(x, idx))
            encoded_eval_dataset = eval_dataset.map(lambda x: encode_labels(x, idx)) #todo: TO MAKE SURE THE LABELS ARE OK AT THE TEST DATASET.
            for inner_model in base_and_pt_models: #runs through the pt_model and rd_pt_model
                tokenized_train_dataset = encoded_train_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)
                tokenized_eval_dataset = encoded_eval_dataset.map(lambda x: tokenize_function(inner_model["tokenizer"], x), batched=True)
                # Initialize the Trainer
                trainer = Trainer(
                    model=inner_model['model'],
                    args=training_args,
                    train_dataset=tokenized_train_dataset,
                    tokenizer=inner_model['tokenizer'],
                    data_collator=inner_model["data_collator"],
                    compute_metrics=compute_metrics
                )
                print(f"About to start FT model: {inner_model['name']} of type: {inner_model['type']}, with dataset number: {idx}")
                # Train the model
                trainer.train()
                print("FT is completed, the saved model was saved.")

                type = inner_model['type']
                save_directory = f'./Saved_models/fine-tuned/{model_name}_{type}'
                trainer.save_model(save_directory)

                # Load the trained model for evaluation
                ft_model = AutoModelForSequenceClassification.from_pretrained(save_directory)

                # Initialize the Trainer for the evaluation phase
                trainer = Trainer(
                    model=ft_model,
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
                    "results": evaluation_results
                }

                results_file_name = f'{model_name}_{type}.txt'
                results_dir = "./Evaluation_results/FT/"
                results_file_path = os.path.join(results_dir, results_file_name)

                with open(results_file_path, "w") as file:
                    file.write(json.dumps(results_with_model, indent=4))

                print(f"Evaluation results for the model: {model_name} saved to {results_file_name}")