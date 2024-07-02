# ______________________________TASKS__________________________________________________________________
# pre-train corpus(financial). - DONE
# for every thing I use in the project i need documentation, articles better academic(lora etc) - DONE
# we need baselines - using a simple model, what are the evaluation results - DONE
# to change the flow, it should be a model which trains for all the datasets and then test on a specific test set. RN it doesnt work this way - DONE
# to clean the directory - DONE
# to record everything using GIT + a word diary - DONE.
# to take just the headlines from fiqa and not the tweets - DONE
# to check it even works, maybe the model not even trains! to make sanity checks - DONE
# to check which dataset should be the test set - DONE
from csv import excel

# to create the back-translated datasets and FT the pre-trained models over all the datasets.

# to create random seeds for each run, and see if i get the same errors for some different seeds. this will get me the real errors and not some random errors to analyse.
# to check which datasets are irrelevant to the test set, meaning their FT doesn't improve results.
# error analysis - out of the errors to understand what's the problem of the model, where are the mistakes and improve it according to those mistakes, or to set-up interpretability, to explain why we have those mistakes.
# to be creative and think about ways to improve results based on the errors, to think about a way to give the model not just text, but another features.
# _____________________________________________________________________________________________________________________________________
# Evaluation Hyper-Parameters: sentiment_threshold.
# ______________________________________________________________________________________________________________________
# IF I WANT TO LOAD THE SAVED MODEL I DO IT THIS WAY:
# save_directory = "./saved_model"
# loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
# loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
# ____________________________________________________________________________________
from translate import Translator
import os
from transformers import DataCollatorForLanguageModeling
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import re
from peft import LoraConfig, TaskType, get_peft_model
import json
import pandas as pd
from datasets import load_metric

#

# Make sure a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hyper_params_bank = {"learning_rate_model0": [1e-5,3e-5, 5e-5], "learning_rate_model1":[2e-5,3.5e-5, 5e-5], "learning_rate_model2":[1e-5,3e-5, 4.5e-5, 6e-5], "train_batch_size":[8, 16, 32, 64], "weight_decay": [0.001, 0.01, 0.1], "eval_batch_size":[8, 16, 32, 64], "mlm_probability":[0.1, 0.15, 0.2],
                "pre_train_batch_size":[8, 16, 32, 64], "lora_rank":[4, 8, 16], "lora_alphas_to_multiply_by_rank":[1, 1.5, 2], "lora_dropout":[0.0, 0.05, 0.1, 0.2]}

SEED = 1694
np.random.seed(SEED)
torch.manual_seed(SEED)

model0 = {"tokenizer": "FacebookAI/roberta-base",
          "model": AutoModelForSequenceClassification.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis').to(device),
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}#distilroberta-FT-financial-news-sentiment-analysis
model1 = {"tokenizer": "KernAI/stock-news-distilbert",
          "model": AutoModelForSequenceClassification.from_pretrained('KernAI/stock-news-distilbert', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('KernAI/stock-news-distilbert'),
          "name": "stock-news-distilbert"}#stock-news-distilbert
model2 = {"tokenizer": "bert-base-uncased",
          "model": AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3).to(device),
          "model_for_PT": AutoModelForMaskedLM.from_pretrained('ProsusAI/finbert').to(device),
          "name": "Finbert"}#FinBert
models = [model0, model1, model2]

PT_model0 = {"save_directory": "./Saved_models/pre_trained/distilroberta-finetuned-financial-news-sentiment-analysis",
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}
PT_model1 = {"save_directory": "./Saved_models/pre_trained/stock-news-distilbert",
          "name": "stock-news-distilbert"}
PT_model2 = {"save_directory": "./Saved_models/pre_trained/Finbert",
          "name": "Finbert"}
PT_models = [PT_model0, PT_model1, PT_model2]

PT_FT_model0 = {"save_directory": "./Saved_models/pre_trained/distilroberta-finetuned-financial-news-sentiment-analysis",
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}

LORA_FLAG = 0
PRE_TRAIN_FLAG = 1
NUM_TRAIN_EPOCH = 3
NUM_DATASETS = 4 #excludes the back-translated datasets

# LORA:
lora_rank = [4, 8, 16]
lora_alpha = lora_rank * 2
# lora_alphas = lora_rank * [1, 1.5, 2]
lora_dropout = [0.0, 0.05, 0.1, 0.2]
idx_lora = 0
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_rank[idx_lora], lora_alpha=lora_alpha[idx_lora],
                         lora_dropout=lora_dropout[idx_lora])

FPB = load_dataset("financial_phrasebank", 'sentences_75agree')['train'].rename_column('sentence', 'text')
train_FPB, test_FPB = FPB.train_test_split(test_size=0.3, seed=SEED).values()

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
    do_eval=True
)
# clean the URLs from fiqa dataset and the "user" word from stock-market sentiment dataset
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

# Receive a sentence and back translate it from French
def back_translate(sentence):
    src_lang = 'en'
    tgt_lang = 'fr'
    translator_to_tgt = Translator(from_lang=src_lang, to_lang=tgt_lang)
    translator_to_src = Translator(from_lang=tgt_lang, to_lang=src_lang)

    # Translate to the chosen language
    translated_text = translator_to_tgt.translate(sentence)

    # Translate back to the original language
    back_translated_text = translator_to_src.translate(translated_text)

    return back_translated_text

# creates the back_translated datasets
def create_extended_datasets():
    for idx in range(0,NUM_DATASETS):
        dataset = get_dataset(idx)

        def back_translate_example(example):
            example['text'] = back_translate(example['text'])
            return example

        translated_dataset = dataset.map(lambda example: back_translate_example(example))

        save_dataset_directory = "Data/"
        os.makedirs(save_dataset_directory, exist_ok=True)

        csv_filename = os.path.join(save_dataset_directory, f"translated_dataset_{idx}.csv")
        translated_dataset.to_csv(csv_filename, index=False)

# Tokenize the datasets for each dataset
def tokenize_function(tokenizer, examples):
    output = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    return output

def encode_labels(example, ds):

    # Encode labels as integers according to: 0-negative, 1-neutral, 2-positive
    # label_dict0 = {0: 1, 1: 2, 2: 0}  # financial-tweets-sentiment
    # label_dict1 = {0: 1, 1: 2, 2: 0}  # synthetic-financial-tweets-sentiment

    label_dict2 = {1:2, -1:0}  # Stock-Market Sentiment Dataset
    if ds == 0:  #fiqa-sentiment-classification
        if example['score'] <= -0.4:
            example['label'] = 0
        elif example['score'] <= 0.5:
            example['label'] = 1
        else:
            example['label'] = 2
    elif ds == 2:  #Stock-Market Sentiment Dataset
        example['label'] = label_dict2[example['label']]
    return example

def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    precision_metric = load_metric("precision")
    recall_metric = load_metric("recall")
    f1_metric = load_metric("f1")

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

def get_dataset(idx):
    if idx == 0: #fiqa-sentiment-classification
        train_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='train').rename_column('sentence', 'text')
        valid_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='valid').rename_column('sentence', 'text')
        test_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='test').rename_column('sentence', 'text')
        concatenate_dataset= concatenate_datasets([train_dataset, valid_dataset, test_dataset])
        dataset = concatenate_dataset.filter(lambda example: example['type'] == 'headline')
        dataset = clean_dataset(dataset, idx)
    elif idx == 1: #financial_phrasebank_75_agree
        dataset = train_FPB
    elif idx == 2: #Stock-Market Sentiment Dataset
        df = pd.read_csv('Data/Stock-Market Sentiment Dataset.csv')
        df.rename(columns={'Text': 'text', 'Sentiment': 'label'}, inplace=True)
        dataset = Dataset.from_pandas(df)
        dataset = clean_dataset(dataset, idx)
    elif idx == 3: #Aspect based Sentiment Analysis for Financial News
        df = pd.read_csv('Data/Processed_Financial_News.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 4: #Back-Translated fiqa-sentiment-classification
        df = pd.read_csv('Data/translated_dataset_0.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 5: #Back-Translated financial_phrasebank_75_agree
        df = pd.read_csv('Data/translated_dataset_1.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 6: #Back-Translated Stock-Market Sentiment Dataset
        df = pd.read_csv('Data/translated_dataset_2.csv')
        dataset = Dataset.from_pandas(df)
    else: #Back-Translated Aspect based Sentiment Analysis for Financial News
        df = pd.read_csv('Data/translated_dataset_3.csv')
        dataset = Dataset.from_pandas(df)
    return dataset

def tokenize_pre_train(tokenizer, example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

# loading the datasets and removing unnecessary cols
# NOT SURE ITS REALLY NECESSARY TO REMOVE COLS
def prepare_for_pre_train():

    # Financial articles datasets for pre-train the model
    pretrain_dataset0 = load_dataset('ashraq/financial-news-articles').remove_columns(['title', 'url'])['train']
    pretrain_dataset1 = load_dataset('BEE-spoke-data/financial-news-articles-filtered').remove_columns(['title', 'url', 'word_count'])['train']
    pretrain_dataset2 = load_dataset('Lettria/financial-articles').remove_columns(['source_name', 'url', 'origin']).rename_column('content', 'text')['train']

    pre_train_datasets = [pretrain_dataset0, pretrain_dataset1, pretrain_dataset2]

    concatenated_dataset = concatenate_datasets(pre_train_datasets)

    # Convert the dataset to a pandas DataFrame
    df = concatenated_dataset.to_pandas()
    # drops the duplicate text from the concatenate dataset
    df = df.drop_duplicates(subset=['text'])
    # Convert the DataFrame back to a Hugging Face Dataset
    cleaned_dataset = Dataset.from_pandas(df)

    return cleaned_dataset

# pre-train the model over 3 unlabeled corpus
def pre_train(model, pre_train_dataset):
    model_name = model['name']
    tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    tokenized_dataset = pre_train_dataset.map(lambda x: tokenize_pre_train(tokenizer, x), batched=True)

    pre_training_args = TrainingArguments(
        output_dir='./preTrain_checkpoints',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
    )

    model = model["model_for_PT"]

    trainer = Trainer(
        model=model,
        args=pre_training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    save_directory = './Saved_models/pre_trained/' + model_name
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f" {model_name} has been pre-trained")

def train_test_all_models(NUM_TRAIN_EPOCH):

    if(PRE_TRAIN_FLAG):
        results_dir = "./Evaluation_results/PT + FT/"
        save_directory = "./Saved_models/PT + FT/" #need to add the model_name
    else:
        results_dir = "./Evaluation_results/FT/"
        save_directory = "./Saved_models/FT/"

    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    test_dataset = test_FPB

    for model in PT_models:
        if (LORA_FLAG):
            chosen_model = get_peft_model(model["model"], lora_config)  # applying LORA
        else:
            # chosen_model = model["model"]
            chosen_model = AutoModelForSequenceClassification.from_pretrained(model["save_directory"])# edit for the PT+FT
        model_name = model["name"]
        # tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        tokenizer = AutoTokenizer.from_pretrained(model["save_directory"])# edit for the PT+FT
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
        tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)


        for idx in range(0, NUM_DATASETS * 2): #includes the Back-Translated DS
            dataset = get_dataset(idx)
            encoded_dataset = dataset.map(lambda x: encode_labels(x, idx))
            tokenized_train_dataset = encoded_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

            # Initialize the Trainer
            trainer = Trainer(
                model=chosen_model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            print(f"About to start FT model: {model_name}, with dataset number: {idx}")
            # Train the model
            trainer.train()

        print("FT is completed, the saved model was saved.")
        # END OF TRAINING

        trainer.save_model(save_directory+"/"+model_name)

        # Initialize the Trainer for the evaluation phase
        trainer = Trainer(
            model=chosen_model,
            args=evaluation_args,
            eval_dataset=tokenized_test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        results_with_model = {
            "Lora YES/NO": LORA_FLAG,
            "Pre-Train YES/NO": PRE_TRAIN_FLAG,
            "model_name": model_name,
            "results": trainer.evaluate()
        }

        results_file_name = model_name + ".txt"
        results_file_path = os.path.join(results_dir, results_file_name)

        with open(results_file_path, "w") as file:
            file.write(json.dumps(results_with_model, indent=4))

        print(f"Evaluation results for the model: {model_name} saved to {results_file_name}")


# train_test_all_models(NUM_TRAIN_EPOCH)
create_extended_datasets()
def main():
    if(PRE_TRAIN_FLAG):
        pre_train_dataset = prepare_for_pre_train()
        for model in models:
            pre_train(model, pre_train_dataset)
    train_test_all_models(NUM_TRAIN_EPOCH)



