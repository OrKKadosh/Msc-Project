import os
import random
from copy import deepcopy

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from nltk import sent_tokenize
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, LlamaForCausalLM, AutoTokenizer, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer

print("Running pre_training.py")

def convert_labels_to_int(example):
    # Convert the labels to integers
    example['label'] = int(example['label'])
    return example
eval_consent_75_df = pd.read_csv('Data/test_datasets/split_eval_test/consent_75_eval.csv').apply(convert_labels_to_int, axis=1)
eval_dataset = Dataset.from_pandas(eval_consent_75_df)


# Make sure a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# base_model0 = {"tokenizer": "FacebookAI/roberta-base",
#           "model": AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
#           "model_for_PT": 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
#           "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}#distilroberta-FT-financial-news-sentiment-analysis
# base_model1 = {"tokenizer": "KernAI/stock-news-distilbert",
#           "model": AutoModelForSequenceClassification.from_pretrained('KernAI/stock-news-distilbert', num_labels=3).to(device),
#           "model_for_PT": 'KernAI/stock-news-distilbert',
#           "name": "stock-news-distilbert"}#stock-news-distilbert
# base_model2 = {"tokenizer": "bert-base-uncased",
#           "model": AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3).to(device),
#           "model_for_PT": 'ProsusAI/finbert',
#           "name": "Finbert"}#FinBert
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

# base_models = [base_model0, base_model1, base_model2, base_model4] #skipped training finGPT for now.
base_models = [base_model4] #skipped training finGPT for now.


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


def random_deletion(dataset):
    def rd(example):
        # Split the text into sentences
        sentences = sent_tokenize(example['text'])

        modified_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 1:  # Only delete if the sentence has more than 1 word
                deletion_idx = random.randint(0, len(words) - 1)
                del words[deletion_idx]  # Remove one word from the sentence
            modified_sentences.append(' '.join(words))  # Rebuild the sentence

        # Join the modified sentences back together into a single string
        example['text'] = ' '.join(modified_sentences)
        return example

    # Apply the function to each example in the dataset
    RID_dataset = dataset.map(lambda example: rd(example))
    return RID_dataset

def get_pretrain_dataset():
    # Financial articles datasets for pre-train the model
    # pretrain_dataset = load_dataset('Lettria/financial-articles').remove_columns(['source_name', 'url', 'origin']).rename_column('content', 'text')['train']
    pretrain_dataset = load_dataset('BEE-spoke-data/financial-news-articles-filtered').remove_columns(['title', 'url', 'word_count'])['train']


    # _, pretrain_dataset = pretrain_dataset.train_test_split(0.3, seed=1694).values()

    # RD_pretrain_dataset = random_deletion(pretrain_dataset)

    # return pretrain_dataset, RD_pretrain_dataset
    return pretrain_dataset


# def pre_train():
#
#     print("start pre-training")
#     pretrain_dataset, RD_pretrain_dataset = get_pretrain_dataset()
#
#     for model in base_models:
#
#         model_name = model['name']
#         tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
#         data_collator = DataCollatorForLanguageModeling(
#             tokenizer=tokenizer,
#             mlm=True,
#             mlm_probability=0.15
#         )
#         print("starts tokenizing first dataset")
#         tokenized_pretrain_dataset = pretrain_dataset.map(lambda x: tokenize_pre_train(tokenizer, x), batched=True)
#         print("starts tokenizing second dataset")
#         tokenized_rd_pretrain_dataset = RD_pretrain_dataset.map(lambda x: tokenize_pre_train(tokenizer, x),
#                                                            batched=True)
#
#         pre_training_args = TrainingArguments(
#             output_dir='./preTrain_checkpoints',
#             overwrite_output_dir=True,
#             num_train_epochs=3,
#             per_device_train_batch_size=8,
#             save_steps=10_000,
#         )
#
#         rd_pre_training_args = TrainingArguments(
#             output_dir='./rd_preTrain_checkpoints',
#             overwrite_output_dir=True,
#             num_train_epochs=3,
#             per_device_train_batch_size=8,
#         )
#
#         model_for_pt = model["model_for_PT"]
#         model_for_rd_pt = model["model_for_PT"]
#
#         # Handle models that require different training approaches
#         pretrain_trainer = Trainer(
#             model=model_for_pt,
#             args=pre_training_args,
#             train_dataset=tokenized_pretrain_dataset,
#             tokenizer=tokenizer,
#             data_collator=data_collator,
#         )
#
#         rd_pretrain_trainer = Trainer(
#             model=model_for_rd_pt,
#             args=rd_pre_training_args,
#             train_dataset=tokenized_rd_pretrain_dataset,
#             tokenizer=tokenizer,
#             data_collator=data_collator,
#         )
#         print(f"starts training model {model_name} with pre-train dataset")
#         # Pre-train the model
#         pretrain_trainer.train()
#         pretrain_save_directory = f'./Saved_models/pre_trained/Pre-Trained_{model_name}'
#         os.makedirs(pretrain_save_directory, exist_ok=True)
#         print(f"about to save{model_name} model to {pretrain_save_directory}")
#         model_for_pt.save_pretrained(pretrain_save_directory)
#         tokenizer.save_pretrained(pretrain_save_directory)
#         print(f"model {model_name} has been saved to {pretrain_save_directory}")
#
#
#         print(f"starts training model {model_name} with rd_pre-train dataset")
#         rd_pretrain_trainer.train()
#         rd_pretrain_save_directory = f'./Saved_models/pre_trained/Pre-Trained+RD_{model_name}'
#         os.makedirs(rd_pretrain_save_directory, exist_ok=True)
#         print(f"about to save{model_name} model to {rd_pretrain_save_directory}")
#         model_for_rd_pt.save_pretrained(rd_pretrain_save_directory)
#         tokenizer.save_pretrained(rd_pretrain_save_directory)
#         print(f"model {model_name} has been saved to {rd_pretrain_save_directory}")
#
#         print(f"{model_name} has been completed pre-training")
#
#
# pre_train()

# ////////////////////// END //////////////////////

def compute_metrics(eval_pred):
    # Unpack predictions and labels from the evaluation results
    logits, labels = eval_pred
    # Convert logits to predicted class indices by taking the argmax across the last dimension
    predictions = np.argmax(logits, axis=-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate precision, recall, and F1 score with macro averaging
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

    # Return the metrics in a dictionary format expected by Trainer
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
def pre_train():
    print("start pre-training at pre_training.py")
    # pretrain_dataset, RD_pretrain_dataset = get_pretrain_dataset()
    pretrain_dataset = get_pretrain_dataset()


    for model in base_models:

        model_name = model['name']
        tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_pre_train(tokenizer, x),batched=True)  # Tokenize eval

        tokenized_pretrain_dataset = pretrain_dataset.map(lambda x: tokenize_pre_train(tokenizer, x), batched=True)
        # tokenized_rd_pretrain_dataset = RD_pretrain_dataset.map(lambda x: tokenize_pre_train(tokenizer, x),batched=True)

        pretrain_output_dir = f'./preTrain_checkpoints/{model_name}'
        # rd_pretrain_output_dir = f'./rd_preTrain_checkpoints_new/{model_name}'

        # pre_training_args = TrainingArguments(
        #     output_dir=pretrain_output_dir,
        #     overwrite_output_dir=True,
        #     num_train_epochs=3,
        #     per_device_train_batch_size=8,
        #     save_steps=10_000,
        #     save_total_limit=2,
        #     logging_dir='./logs',
        # )

        pre_training_args = TrainingArguments(
            output_dir=pretrain_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir='./logs',
        )

        # rd_pre_training_args = TrainingArguments(
        #     output_dir=rd_pretrain_output_dir,
        #     overwrite_output_dir=True,
        #     num_train_epochs=3,
        #     per_device_train_batch_size=8,
        #     save_steps=10_000,
        #     save_total_limit=2,
        #     logging_dir='./logs',
        # )

        model_for_pt = AutoModelForMaskedLM.from_pretrained(model["model_for_PT"]).to(device)
        # model_for_rd_pt = AutoModelForMaskedLM.from_pretrained(model["model_for_PT"]).to(device)

        pretrain_trainer = Trainer(
            model=model_for_pt,
            args=pre_training_args,
            train_dataset=tokenized_pretrain_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )


        # rd_pretrain_trainer = Trainer(
        #     model=model_for_rd_pt,
        #     args=rd_pre_training_args,
        #     train_dataset=tokenized_rd_pretrain_dataset,
        #     tokenizer=tokenizer,
        #     data_collator=data_collator,
        # )

        pretrain_trainer.train()

        # Save the final model after pre-training on the first dataset
        pretrain_save_directory = f'./Saved_models/pre_trained_BEE-spoke/Pre-Trained_{model_name}'
        os.makedirs(pretrain_save_directory, exist_ok=True)
        model_for_pt.save_pretrained(pretrain_save_directory)
        tokenizer.save_pretrained(pretrain_save_directory)
        print(f"model {model_name} has been saved to {pretrain_save_directory}")

        # rd_pretrain_trainer.train()

        # Save the final model after pre-training on the second dataset
        # rd_pretrain_save_directory = f'./Saved_models/pre_trained_BEE-spoke/Pre-Trained+RD_{model_name}'
        # os.makedirs(rd_pretrain_save_directory, exist_ok=True)
        # model_for_rd_pt.save_pretrained(rd_pretrain_save_directory)
        # tokenizer.save_pretrained(rd_pretrain_save_directory)
        # print(f"model {model_name} has been saved to {rd_pretrain_save_directory}")
        #
        # print(f"{model_name} has completed pre-training")

pre_train()




