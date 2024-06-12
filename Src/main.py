# ______________________________TASKS__________________________________________________________________
# pre-train corpus(financial). - DONE
# for every thing I use in the project i need documentation, articles better academic(lora etc) - DONE
# we need baselines - using a simple model, what are the evaluation results - DONE
# to change the flow, it should be a model which trains for all the datasets and then test on a specific test set. RN it doesnt work this way - DONE

# to check which dataset should be the test set
# to check which datasets are irrelevant to the test set, meaning their FT doesn't improve results
# to clean the directory.
# to make a logs folder.
# error analysis - out of the errors to understand what's the problem of the model, where are the mistakes and improve it according to those mistakes, or to set-up interpretability, to explain why we have those mistakes.
# to record everything using GIT + a word diary.
# to be creative and think about ways to improve results based on the errors, to think about a way to give the model not just text, but another features.
# to create random seeds for each run, and see if i get the same errors for some different seeds. this will get me the real errors and not some random errors to analyse.
# to check it even works, maybe the model not even trains! to make sanity checks
# _____________________________________________________________________________________________________________________________________
# general Hyper-Params: epochs, learning_rate, per_device_train_batch_size, per_device_eval_batch_size, num_train_epochs, weight_decay.
# Evaluation Hyper-Parameters: sentiment_threshold.
# LORA - Hyper-Params: r= [4, 8, 16], alpha = lora_rank * 2, dropout = [0.1, 0.2, 0.3, 0.4, 0.5], layers_to_enable_lora.
# ______________________________________________________________________________________________________________________
# IF I WANT TO LOAD THE SAVED MODEL I DO IT THIS WAY:
# save_directory = "./saved_model"
# loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
# loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
# ____________________________________________________________________________________

import os
from transformers import DataCollatorForLanguageModeling
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
import json
from datasets import load_metric

hyper_params_bank = {"learning_rate_model0": [1e-5,3e-5, 5e-5], "learning_rate_model1":[2e-5,3.5e-5, 5e-5], "learning_rate_model2":[1e-5,3e-5, 4.5e-5, 6e-5], "train_batch_size":[8, 16, 32, 64], "weight_decay": [0.001, 0.01, 0.1], "eval_batch_size":[8, 16, 32, 64], "mlm_probability":[0.1, 0.15, 0.2],
                "pre_train_batch_size":[8, 16, 32, 64], "lora_rank":[4, 8, 16], "lora_alphas_to_multiply_by_rank":[1, 1.5, 2], "lora_dropout":[0.0, 0.05, 0.1, 0.2]}

SEED = 1694
np.random.seed(SEED)
torch.manual_seed(SEED)

# Make sure a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def check_checkpoints():
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": (preds == p.label_ids).mean()}

    checkpoint_dir1 = "./train_checkpoints/checkpoint-303"
    checkpoint_dir2 = "./train_checkpoints/checkpoint-606"
    checkpoint_dir3 = "./train_checkpoints/checkpoint-909"
    model1 = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir1)
    model2 = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir2)
    model3 = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir3)
    models = [model1, model2, model3]

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    tokenized_test =test_FPB.map(lambda x: tokenize_function(tokenizer, x, 3), batched=True)


    output_dir = "./results/checkpoints"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=1000,  # Save checkpoint every 1000 steps
        save_total_limit=2  # Limit the total amount of checkpoints
    )

    for model in models:
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_dataset=tokenized_test,
            compute_metrics=compute_metrics
        )

        print(trainer.evaluate())

# get the initialized tokenizer
def get_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


# Tokenize the datasets for each dataset
def tokenize_function(tokenizer, examples, ds):
    if ds == 0:  #financial-tweets-sentiment
        return tokenizer(examples['tweet'], padding='max_length', truncation=True, max_length=256)
    elif ds == 1:  #synthetic-financial-tweets-sentiment
        return tokenizer(examples['tweet'], padding='max_length', truncation=True, max_length=256)
    elif ds == 2:  #fiqa-sentiment-classification
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=256)
    elif ds == 3:  #financial_phrasebank_75_agree
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=256)
    elif ds == 4:  #twitter-financial-news-sentiment
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)


def encode_labels(example, ds):

    # Encode labels as integers according to: 0-negative, 1-neutral, 2-positive
    label_dict0 = {0: 1, 1: 2, 2: 0}  # financial-tweets-sentiment
    label_dict1 = {0: 1, 1: 2, 2: 0}  # synthetic-financial-tweets-sentiment
    # label_dict3 = {0: 0, 1: 1, 2: 2}  # financial_phrasebank
    label_dict4 = {0: 0, 1: 2, 2: 1}  # twitter-financial-news-sentiment

    if ds == 0:  #financial-tweets-sentiment
        example['label'] = label_dict0[example['sentiment']]
    elif ds == 1:  #synthetic-financial-tweets-sentiment
        example['label'] = label_dict1[example['sentiment']]
    elif ds == 2:  #fiqa-sentiment-classification
        if example['score'] <= -0.4:
            example['label'] = 0
        elif example['score'] <= 0.5:
            example['label'] = 1
        else:
            example['label'] = 2
    elif ds == 3:  #financial_phrasebank
        # example['label'] = label_dict3[example['label']]
        return example #no mapping in this case
    elif ds == 4:  #twitter-financial-news-sentiment
        example['label'] = label_dict4[example['label']]
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
    if idx == 0:
        dataset = load_dataset("TimKoornstra/financial-tweets-sentiment")
        dataset = dataset['train']
    elif idx == 1:
        dataset = load_dataset("TimKoornstra/synthetic-financial-tweets-sentiment", ignore_verifications=True)
        dataset = dataset['train']
    elif idx == 2:
        train_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='train')
        valid_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='valid')
        test_dataset = load_dataset("ChanceFocus/fiqa-sentiment-classification", split='test')
        dataset = concatenate_datasets([train_dataset, valid_dataset, test_dataset])
    elif idx == 3:
        dataset = train_FPB
    else:
        train_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split='train')
        valid_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split='validation')
        dataset = concatenate_datasets([train_dataset, valid_dataset])
    return dataset


# Load the model and send it to the GPU
model0 = {"tokenizer": "FacebookAI/roberta-base", "model": AutoModelForSequenceClassification.from_pretrained(
    'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}
model1 = {"tokenizer": "distilbert/distilbert-base-uncased",
          "model": AutoModelForSequenceClassification.from_pretrained('KernAI/stock-news-distilbert', num_labels=3).to(
              device), "name": "stock-news-distilbert"}
model2 = {"tokenizer": "FacebookAI/roberta-base", "model": AutoModelForSequenceClassification.from_pretrained(
    'cardiffnlp/twitter-roberta-base-sentiment-latest', num_labels=3).to(device),
          "name": "twitter-roberta-base-sentiment-latest"}

# models = [model0, model1, model2]
models = [model0]

# LORA:
lora_rank = [4, 8, 16]
lora_alpha = lora_rank * 2
# lora_alphas = lora_rank * [1, 1.5, 2]
lora_dropout = [0.0, 0.05, 0.1, 0.2]
idx_lora = 0
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_rank[idx_lora], lora_alpha=lora_alpha[idx_lora],
                         lora_dropout=lora_dropout[idx_lora])

FPB = load_dataset("financial_phrasebank", 'sentences_75agree')['train']
train_FPB, test_FPB = FPB.train_test_split(test_size=0.3, seed=SEED).values()


def tokenize_pre_train(example, tokenizer):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)


def pre_train(model):
    # Financial articles datasets for pre-train the model
    pretrain_dataset1 = load_dataset('ashraq/financial-news-articles')
    pretrain_dataset2 = load_dataset('BEE-spoke-data/financial-news-articles-filtered')
    pretrain_dataset3 = load_dataset('Lettria/financial-articles')

    tokenizer = get_tokenizer(model["tokenizer"])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    tokenized_dataset1 = pretrain_dataset1.map.map(lambda x: tokenize_pre_train(tokenizer, x), batched=True)
    tokenized_dataset2 = pretrain_dataset2.map.map(lambda x: tokenize_pre_train(tokenizer, x), batched=True)
    tokenized_dataset3 = pretrain_dataset3.map.map(lambda x: tokenize_pre_train(tokenizer, x), batched=True)

    combined_train_dataset = concatenate_datasets(
        [tokenized_dataset1['train'], tokenized_dataset2['train'], tokenized_dataset3['train']])

    training_args = TrainingArguments(
        output_dir='./preTrain_checkpoints',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    return model

def train_test_all_models(NUM_TRAIN_EPOCH):



    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./train_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=NUM_TRAIN_EPOCH,
        weight_decay=0.01,
        save_strategy="epoch",
        save_steps=500,
        save_total_limit=2
    )
    # Set up evaluation arguments
    evaluation_args = TrainingArguments(
        output_dir = "./eval_checkpoints",
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        do_eval=True
    )

    NUM_DATASETS = 5
    LORA_FLAG = 0
    PRE_TRAIN_FLAG = 0

    test_dataset = test_FPB

    for model in models:

        model_name = model["name"]
        tokenizer = get_tokenizer(model["tokenizer"])
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
        tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(tokenizer, x, 3), batched=True)

        if (LORA_FLAG):
            chosen_model = get_peft_model(model["model"], lora_config)  # applying LORA
        else:
            chosen_model = model["model"]

        if (PRE_TRAIN_FLAG):
            chosen_model = pre_train(chosen_model)

        for idx in range(3, NUM_DATASETS):
            dataset = get_dataset(idx)
            encoded_dataset = dataset.map(lambda x: encode_labels(x, idx))
            tokenized_train_dataset = encoded_dataset.map(lambda x: tokenize_function(tokenizer, x, idx), batched=True)

            # Initialize the data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

            # Initialize the Trainer
            trainer = Trainer(
                model=chosen_model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            print(f"About to train model: {model_name}, with dataset number: {idx}")
            # Train the model
            trainer.train()

        print("Training is completed!")
        # END OF TRAINING

        save_directory = "./Saved_models/" + model_name
        os.makedirs(save_directory, exist_ok=True)
        trainer.save_model(save_directory)

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

        results_dir = "./Evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        results_file_name = model_name + "_results.txt"
        results_file_path = os.path.join(results_dir, results_file_name)

        with open(results_file_path, "w") as file:
            file.write(json.dumps(results_with_model, indent=4))

        print(f"Evaluation results for the model: {model_name} saved to {results_file_name}")

# train_test_all_models(3)





