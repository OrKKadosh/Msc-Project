import os

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, LlamaForCausalLM, AutoTokenizer, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer

# Make sure a GPU is available
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
base_model3 = {
    "tokenizer": "NousResearch/Llama-2-13b-hf",
    "model": PeftModel.from_pretrained(
        LlamaForCausalLM.from_pretrained(
            "NousResearch/Llama-2-13b-hf", trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
        ),
        "FinGPT/fingpt-sentiment_llama2-13b_lora"
    ),
    "model_for_PT": PeftModel.from_pretrained(
        LlamaForCausalLM.from_pretrained(
            "NousResearch/Llama-2-13b-hf", trust_remote_code=True, device_map="cuda:0", load_in_8bit=True
        ),
        "FinGPT/fingpt-sentiment_llama2-13b_lora"
    ),
    "name": "FinGPT"
} #FinGPT

base_model4 = {
    "tokenizer": "SALT-NLP/FLANG-ELECTRA",
    "model": AutoModelForSequenceClassification.from_pretrained("SALT-NLP/FLANG-ELECTRA", num_labels=3).to(device),
    "model_for_PT": AutoModelForMaskedLM.from_pretrained("SALT-NLP/FLANG-ELECTRA").to(device),
    "name": "FLANG-ELECTRA"
}#FLANG-ELECTRA
# base_models = [base_model0, base_model1, base_model2, base_model3, base_model4]
base_models = [base_model0]


def tokenize_pre_train(tokenizer, example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

def clean_csv_file(input_path, output_path):
    def clean_row(row):
        # Replace newlines inside quoted strings with spaces
        return row.replace('\n', ' ').replace('\r', '')

    # Read the file line by line, clean rows
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = [clean_row(line) for line in lines]

    # Write cleaned lines to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)


def pre_train():
    print("start")
    for model in base_models:
        # Clean the CSV files first
        clean_csv_file('Data/PreTrain/pretrain_dataset_cleaned.csv', 'Data/PreTrain/pretrain_dataset_cleaned_fixed.csv')
        clean_csv_file('Data/PreTrain+RD/rd_pretrain_dataset_cleaned.csv',
                       'Data/PreTrain+RD/rd_pretrain_dataset_cleaned_fixed.csv')

        print("passed cleaning csv")
        # Now read the cleaned CSVs
        pre_train_df = pd.read_csv('Data/PreTrain/pretrain_dataset_cleaned_fixed.csv')
        print("read first file, HURRAYYY")
        rd_pre_train_df = pd.read_csv('Data/PreTrain+RD/rd_pretrain_dataset_cleaned_fixed.csv')
        print("read second file, HURRAYYY")
        pre_train_dataset = Dataset.from_pandas(pre_train_df)
        rd_pre_train_dataset = Dataset.from_pandas(rd_pre_train_df)

        model_name = model['name']
        tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        tokenized_pretrain_dataset = pre_train_dataset.map(lambda x: tokenize_pre_train(tokenizer, x), batched=True)
        tokenized_rd_pretrain_dataset = rd_pre_train_dataset.map(lambda x: tokenize_pre_train(tokenizer, x),
                                                                 batched=True)

        pre_training_args = TrainingArguments(
            output_dir='./preTrain_checkpoints',
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=10_000,
        )

        rd_pre_training_args = TrainingArguments(
            output_dir='./rd_preTrain_checkpoints',
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=10_000,
        )

        model_to_train = model["model_for_PT"]

        # Handle models that require different training approaches
        if model_name == "FinGPT":
            pretrain_trainer = Trainer(
                model=model_to_train,
                args=pre_training_args,
                train_dataset=tokenized_pretrain_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            rd_pretrain_trainer = Trainer(
                model=model_to_train,
                args=rd_pre_training_args,
                train_dataset=tokenized_rd_pretrain_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

        else:
            pretrain_trainer = Trainer(
                model=model_to_train,
                args=pre_training_args,
                train_dataset=tokenized_pretrain_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            rd_pretrain_trainer = Trainer(
                model=model_to_train,
                args=rd_pre_training_args,
                train_dataset=tokenized_rd_pretrain_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

        # Pre-train the model
        pretrain_trainer.train()
        rd_pretrain_trainer.train()

        # Create directories for saving the models
        pretrain_save_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
        rd_pretrain_save_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'

        os.makedirs(pretrain_save_directory, exist_ok=True)
        os.makedirs(rd_pretrain_save_directory, exist_ok=True)

        print("about to save")

        # Save the model and tokenizer
        model_to_train.save_pretrained(pretrain_save_directory)
        model_to_train.save_pretrained(rd_pretrain_save_directory)
        tokenizer.save_pretrained(pretrain_save_directory)
        tokenizer.save_pretrained(rd_pretrain_save_directory)

        print(f"{model_name} has been pre-trained")

pre_train()