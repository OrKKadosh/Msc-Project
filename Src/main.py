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

# to create the back-translated datasets and FT the pre-trained models over all the datasets.

# to create random seeds for each run, and see if i get the same errors for some different seeds. this will get me the real errors and not some random errors to analyse.
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
from csv import excel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import wandb
from nltk.corpus import sentiwordnet as swn
nltk.download('sentiwordnet')
nltk.download('wordnet')
import os
from transformers import DataCollatorForLanguageModeling, LlamaForCausalLM
import torch
import time
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import random
import requests
import re
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import json
import pandas as pd
import evaluate
from google.cloud import translate_v2 as translate


# Ensure the environment variable is set
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/cs_storage/orkados/plucky-mode-428714-b6-8a8d416836b7.json"

# Make sure a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hyper_params_bank = {"learning_rate_model0": [2e-5,3e-5, 5e-5], "learning_rate_model1":[2e-5,3.5e-5, 5e-5], "learning_rate_model2":[1e-5,3e-5, 4.5e-5, 6e-5], "train_batch_size":[8, 16, 32, 64], "weight_decay": [0.001, 0.01, 0.1], "eval_batch_size":[8, 16, 32, 64], "mlm_probability":[0.1, 0.15, 0.2],
                "pre_train_batch_size":[8, 16, 32, 64], "lora_rank":[4, 8, 16], "lora_alphas_to_multiply_by_rank":[1, 1.5, 2], "lora_dropout":[0.0, 0.05, 0.1, 0.2]}

SEED = 1694
np.random.seed(SEED)
torch.manual_seed(SEED)

# Initialize wandb
wandb.init(
    project="your-project-name",
    config={
        "learning_rate": 2e-5,
        "architecture": "BERT",
        "dataset": "Financial Sentiment",
        "epochs": 3,
        "batch_size": 8,
        "seed": SEED,
    }
)


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
    "name": "FinGPT"}#FinGPT
base_model4 = {
    "tokenizer": "SALT-NLP/FLANG-ELECTRA",
    "model": AutoModelForSequenceClassification.from_pretrained("SALT-NLP/FLANG-ELECTRA", num_labels=3).to(device),
    "name": "FLANG-ELECTRA"} #Flang-Electra

base_models = [base_model0, base_model1, base_model2, base_model3, base_model4]

PT_model0 = {"save_directory": "./Saved_models/pre_trained/distilroberta-finetuned-financial-news-sentiment-analysis",
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis_PT"}
PT_model1 = {"save_directory": "./Saved_models/pre_trained/stock-news-distilbert",
          "name": "stock-news-distilbert_PT"}
PT_model2 = {"save_directory": "./Saved_models/pre_trained/Finbert",
          "name": "Finbert_PT"}
PT_models = [PT_model0, PT_model1, PT_model2]

PT_FT_model0 = {"save_directory": "./Saved_models/PT + FT/distilroberta-finetuned-financial-news-sentiment-analysis",
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}
PT_FT_model1 = {"save_directory": "./Saved_models/PT + FT/stock-news-distilbert",
          "name": "stock-news-distilbert"}
PT_FT_model2 = {"save_directory": "./Saved_models/PT + FT/Finbert",
          "name": "Finbert"}
PT_FT_models = [PT_FT_model0, PT_FT_model1, PT_FT_model2]

LORA_FLAG = 0
PRE_TRAIN_FLAG = 1
NUM_TRAIN_EPOCH = 3
NUM_DATASETS = 4 #excludes the back-translated datasets

# LORA: I should try the LORA only for the FinBert since it has ~110 parameters.
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

# Function to process the model outputs
def extract_sentiment_from_generated_text(generated_text):
    # Split the output to find the sentiment after the 'Answer: ' part
    if "Answer: " in generated_text:
        return generated_text.split("Answer: ")[-1].strip().lower()
    return "Error"

# Define a new compute metrics function that works with text generation
def compute_metrics_for_instruction_based(eval_pred):
    # Get the generated texts (instead of logits)
    generated_texts = eval_pred.predictions
    references = eval_pred.label_ids

    # Convert the generated text to labels (0: negative, 1: neutral, 2: positive)
    predictions = []
    for text in generated_texts:
        sentiment = extract_sentiment_from_generated_text(text)
        if sentiment == 'positive':
            predictions.append(2)
        elif sentiment == 'negative':
            predictions.append(0)
        else:
            predictions.append(1)

    # Compute accuracy, precision, recall, and f1
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    precision = precision_metric.compute(predictions=predictions, references=references, average='macro')
    recall = recall_metric.compute(predictions=predictions, references=references, average='macro')
    f1 = f1_metric.compute(predictions=predictions, references=references, average='macro')

    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1']
    }


# Function to create prompts for this specific model (with instruction-based input)
def create_instruction_based_prompt(text):
    prompt = f'''Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.
    Input: {text}
    Answer: '''
    return prompt

def preprocess_instruction_based_dataset(examples, tokenizer):
    # Create the instruction-based prompts
    examples['text'] = [create_instruction_based_prompt(text) for text in examples['text']]
    # Tokenize the instruction-based prompts
    tokens = tokenizer(examples['text'], return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    return tokens


# VADER Sentiment Analyzer
# TODO: DIDNT USE YET
def get_vader_sentiment(text):
    vader_analyzer = SentimentIntensityAnalyzer()
    scores = vader_analyzer.polarity_scores(text)
    return scores


import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Download the necessary NLTK data files
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def create_negation_datasets():

    def process_sentence(example):    # Example: ("i don't like to go swimming") -> ("i do dislike to go swimming")

        def tokenize_sentence(sentence):
            # Tokenize the sentence into words
            tokens = word_tokenize(sentence)
            return tokens

        def get_antonym(word):
            antonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        antonyms.append(lemma.antonyms()[0].name())
            return antonyms[0] if antonyms else word

        def replace_with_opposite(tokens):
            new_tokens = []
            i = 0
            while i < len(tokens):
                if tokens[i] in ["not", "n't"]:
                    if i + 1 < len(tokens):
                        antonym = get_antonym(tokens[i + 1])

                        new_tokens.append(antonym)
                        i += 2  # Skip the "not" or "n't" and the next token
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            return new_tokens

        tokenized_sentence = tokenize_sentence(example['text'])
        modified_tokens = replace_with_opposite(tokenized_sentence)
        example['text'] = ' '.join(modified_tokens)
        return example

    for idx in range(NUM_DATASETS):
        dataset = get_dataset(idx)
        negation_dataset = dataset.map(lambda example : process_sentence(example))

        save_dataset_directory = "Data/Negation/"
        os.makedirs(save_dataset_directory, exist_ok=True)
        csv_filename = os.path.join(save_dataset_directory, f"negation_dataset_{idx}.csv")
        negation_dataset.to_csv(csv_filename, index=False)

# #Example
# sentence = "i don't like Hapoel"
# print(process_sentence(sentence))


# clean the URLs from fiqa dataset and the "user" word from stock-market sentiment dataset
# TODO: DIDNT USE YET
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

#Goes over the FT-datasets, back-translates them and saves them as ./Data/translated_dataset_{idx}.csv
def create_back_translated_datasets():
    request_count = 0

    for idx in range(0, NUM_DATASETS):
        # changes the score to label
        def fix_fiqa_dataset(example):
            if example['score'] <= -0.4:
                example['label'] = 0
            elif example['score'] <= 0.5:
                example['label'] = 1
            else:
                example['label'] = 2
            return example #

        dataset = get_dataset(idx)
        if idx == 0:#change the score to label
            dataset = dataset.map(lambda example: fix_fiqa_dataset(example))
            dataset = dataset.remove_columns(['_id', 'target', 'aspect', 'type'])

        def back_translate_example(example, dataset):

            def back_translate(sentence):
                def translatee(text, source_lang, target_lang):
                    translate_client = translate.Client()
                    result = translate_client.translate(text, source_language=source_lang, target_language=target_lang)
                    return result['translatedText']

                src_lang = 'en'
                tgt_lang = 'fr'

                translated_text = translatee(sentence, src_lang, tgt_lang)
                if translated_text is None:
                    return None

                back_translated_text = translatee(translated_text, tgt_lang, src_lang)
                if back_translated_text is None:
                    return None

                return back_translated_text

            nonlocal request_count
            example['text'] = back_translate(example['text'])

            # request_count += 2  # 2 requests per example (en->fr and fr->en)

            # # Check if 80 requests have been made and pause if necessary
            # if request_count >= 20:
            #     time.sleep(30)
            #     request_count = 0  # Reset request count after sleeping

            return example

        translated_dataset = dataset.map(lambda example: back_translate_example(example, dataset))

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

def get_dataset(idx):
    def clean_text(example, idx):
        if idx in (5,6,7):
            example['text'] = example['text'].replace('&#39;', "'")
        if idx in (5,6):
            # Remove non-English characters using regular expression
            example['text'] = re.sub(r'[^A-Za-z0-9\s.,!?\'\"-]', '', example['text'])
        return example

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
        df = pd.read_csv('Data/back_translated/translated_dataset_0.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 5: #Back-Translated financial_phrasebank_75_agree
        df = pd.read_csv('Data/back_translated/translated_dataset_1.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda example: clean_text(example, idx))
    elif idx == 6: #Back-Translated Stock-Market Sentiment Dataset
        df = pd.read_csv('Data/back_translated/translated_dataset_2.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda example: clean_text(example, idx))
    elif idx == 7: #Back-Translated Aspect based Sentiment Analysis for Financial News
        df = pd.read_csv('Data/back_translated/translated_dataset_3.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda example: clean_text(example, idx))

    elif idx == 8: #negation_fiqa-sentiment-classification
        df = pd.read_csv('Data/negation_dataset_0.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 9: #negation_financial_phrasebank_75_agree
        df = pd.read_csv('Data/negation_dataset_1.csv')
        dataset = Dataset.from_pandas(df)
    elif idx == 10: #negation_Stock-Market Sentiment Dataset
        df = pd.read_csv('Data/negation_dataset_2.csv')
        dataset = Dataset.from_pandas(df)
    else: #idx == 11: #negation_Aspect based Sentiment Analysis for Financial News
        df = pd.read_csv('Data/negation_dataset_3.csv')
        dataset = Dataset.from_pandas(df)
    return dataset

def get_dataset_with_fingpt_dataset(idx):
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
        dataset = train_FPB
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

def tokenize_pre_train(tokenizer, example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

# Randomly deletes a single word from the dataset
def random_deletion(dataset):
    def rd(example):
        words = example['text'].split()
        if len(words) <= 1: # a sentence of one word
            return example
        else:
            deletion_idx = random.randint(0, len(words) - 1)
            del words[deletion_idx]
            example['text'] = ' '.join(words)
            return example

    RID_dataset = dataset.map(lambda example: rd(example))
    return RID_dataset

# loading the datasets and removing unnecessary cols, returns a concatenate dataset of the pre-train datasets
# DIDNT YET PRE-TRAINED THE MODELS OVER THE RANDOMLY DELETED DATASETS
def prepare_for_pre_train():

    # Financial articles datasets for pre-train the model
    pretrain_dataset0 = load_dataset('ashraq/financial-news-articles').remove_columns(['title', 'url'])['train']
    pretrain_dataset1 = load_dataset('BEE-spoke-data/financial-news-articles-filtered').remove_columns(['title', 'url', 'word_count'])['train']
    pretrain_dataset2 = load_dataset('Lettria/financial-articles').remove_columns(['source_name', 'url', 'origin']).rename_column('content', 'text')['train']

    RID_pretrain_dataset0 = random_deletion(pretrain_dataset0)
    RID_pretrain_dataset1 = random_deletion(pretrain_dataset1)
    RID_pretrain_dataset2 = random_deletion(pretrain_dataset2)
    RID_pretrain_datasets = [RID_pretrain_dataset0, RID_pretrain_dataset1, RID_pretrain_dataset2]

    # Use the following code only once, to check the rid_datasets, to see it really randomly deletes a word in the samples
    # save_dataset_directory = "Data/RID_PreTrain/"
    # os.makedirs(save_dataset_directory, exist_ok=True)
    # for idx in range(3):
    #     csv_filename = os.path.join(save_dataset_directory, f"RID_pretrain_dataset_{idx}.csv")
    #     RID_pretrain_datasets[idx].to_csv(csv_filename, index=False)


    pre_train_datasets = [pretrain_dataset0, pretrain_dataset1, pretrain_dataset2, RID_pretrain_dataset0, RID_pretrain_dataset1, RID_pretrain_dataset2]

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

    save_directory = './Saved_models/pre_trained/only RID' + model_name
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f" {model_name} has been pre-trained")

def train_test_all_models(NUM_TRAIN_EPOCH):

    if(PRE_TRAIN_FLAG):
        results_dir = "./Evaluation_results/PT + FT(with BT)/"
        save_directory = "./Saved_models/PT + FT(with BT)/" #need to add the model_name
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
            chosen_model = AutoModelForSequenceClassification.from_pretrained(model["save_directory"])# edited for the PT+FT
        model_name = model["name"]
        # tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
        tokenizer = AutoTokenizer.from_pretrained(model["save_directory"])# edited for the PT+FT
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
                compute_metrics=compute_metrics
            )
            print(f"About to start FT model: {model_name}, with dataset number: {idx}")
            # Train the model
            trainer.train()

            # Log training metrics to wandb
            metrics = trainer.state.log_history[-1]
            wandb.log(metrics)

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

        evaluation_results = trainer.evaluate()

        # Log evaluation metrics to wandb
        wandb.log(evaluation_results)

        results_with_model = {
            "Lora YES/NO": LORA_FLAG,
            "Pre-Train YES/NO": PRE_TRAIN_FLAG,
            "model_name": model_name,
            "results": evaluation_results
        }

        results_file_name = model_name + ".txt"
        results_file_path = os.path.join(results_dir, results_file_name)

        with open(results_file_path, "w") as file:
            file.write(json.dumps(results_with_model, indent=4))

        print(f"Evaluation results for the model: {model_name} saved to {results_file_name}")

# Initialize wandb run
wandb.init(project="your-project-name", config={
    "learning_rate": 2e-5,
    "architecture": "BERT",
    "dataset": "Financial Sentiment",
    "epochs": NUM_TRAIN_EPOCH,
    "batch_size": 8,
    "seed": SEED,
})

train_test_all_models(NUM_TRAIN_EPOCH)

# Finish the wandb run
wandb.finish()

# pre_train_dataset = prepare_for_pre_train()
# for model in base_models:
#     pre_train(model, pre_train_dataset)
# print("ended pre-train the base models")
# for model in PT_models:
#     pre_train(model, pre_train_dataset)

def main():
    print("hey")
    create_negation_datasets()
    if(PRE_TRAIN_FLAG):
        pre_train_dataset = prepare_for_pre_train()
        for model in base_models:
            pre_train(model, pre_train_dataset)
    train_test_all_models(NUM_TRAIN_EPOCH)

