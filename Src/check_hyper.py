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
import itertools


hyper_params_bank = {"learning_rate_model0": [1e-5,3e-5, 5e-5], "learning_rate_model1":[2e-5,3.5e-5, 5e-5], "learning_rate_model2":[1e-5,3e-5, 4.5e-5, 6e-5], "train_batch_size":[8, 16, 32, 64], "weight_decay": [0.001, 0.01, 0.1], "eval_batch_size":[8, 16, 32, 64], "mlm_probability":[0.1, 0.15, 0.2],
                "pre_train_batch_size":[8, 16, 32, 64], "lora_rank":[4, 8, 16], "lora_alphas_to_multiply_by_rank":[1, 1.5, 2], "lora_dropout":[0.0, 0.05, 0.1, 0.2]}
keys, values = zip(*hyper_params_bank.items())
combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]


SEED = 1694
np.random.seed(SEED)
torch.manual_seed(SEED)

# Make sure a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

    best_model = None
    best_hyperparams = None
    best_valid_score = float('-inf')

    NUM_DATASETS = 5
    LORA_FLAG = 1
    PRE_TRAIN_FLAG = 0

    dataset = load_dataset("financial_phrasebank", 'sentences_75agree')['train']
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
    train_valid_dataset = dataset['train'].train_test_split(test_size=0.25, seed=SEED)  # 0.25 * 0.8 = 0.2 of original data
    train_dataset = train_valid_dataset['train']
    valid_dataset = train_valid_dataset['test']
    test_dataset = dataset['test']



    for combination in combinations:

        # LORA:
        lora_rank = combination["lora_rank"]
        lora_alpha = lora_rank * combination["lora_alphas_to_multiply_by_rank"]
        lora_dropout = combination["lora_dropout"]
        idx_lora = 0
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_rank, lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./train_checkpoints",
            learning_rate=combination["learning_rate"],
            per_device_train_batch_size=combination["train_batch_size"],
            num_train_epochs=NUM_TRAIN_EPOCH,
            weight_decay=combination["weight_decay"],
            save_strategy="epoch",
            save_steps=500,
        )
        # Set up evaluation arguments
        evaluation_args = TrainingArguments(
            output_dir="./eval_checkpoints",
            per_device_eval_batch_size=combination["eval_batch_size"],
            logging_dir='./logs',
            do_eval=True
        )


        for model in models:

            model_name = model["name"]
            tokenizer = get_tokenizer(model["tokenizer"])
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
            tokenized_valid_dataset = valid_dataset.map(lambda x: tokenize_function(tokenizer, x, 3), batched=True)

            if (LORA_FLAG):
                chosen_model = get_peft_model(model["model"], lora_config)  # applying LORA
            else:
                chosen_model = model["model"]

            if (PRE_TRAIN_FLAG):
                chosen_model = pre_train(chosen_model)

            for idx in range(0, NUM_DATASETS):
                dataset = get_dataset(idx)
                encoded_dataset = dataset.map(lambda x: encode_labels(x, idx))
                tokenized_train_dataset = encoded_dataset.map(lambda x: tokenize_function(tokenizer, x, idx), batched=True)

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

            save_directory = "./FT_models/" + model_name
            os.makedirs(save_directory, exist_ok=True)
            trainer.save_model(save_directory)

            # Initialize the Trainer for the evaluation phase
            trainer = Trainer(
                model=chosen_model,
                args=evaluation_args,
                eval_dataset=tokenized_valid_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            eval_results = trainer.evaluate()
            valid_score = eval_results['eval_accuracy']

            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_model = chosen_model
                best_hyperparams = combination


            results_with_model = {
                "model_name": model_name,
                "Lora YES/NO": LORA_FLAG,
                "Pre-Train YES/NO": PRE_TRAIN_FLAG,
                "Hyperparams": combination,
                "results": eval_results
            }

            results_dir = "./Evaluation_results"
            os.makedirs(results_dir, exist_ok=True)
            results_file_name = model_name + "_results.txt"
            results_file_path = os.path.join(results_dir, results_file_name)

            with open(results_file_path, "w") as file:
                file.write(json.dumps(results_with_model, indent=4))

            print(f"Evaluation results for the model: {model_name} saved to {results_file_name}")

    print("Best hyperparameters found: ", best_hyperparams)
    print("Training the best model on the combined training and validation sets...")

    combined_train_valid_dataset = concatenate_datasets([train_dataset, valid_dataset])
    tokenized_train_valid_dataset = combined_train_valid_dataset.map(lambda x: tokenize_function(tokenizer, x, 3), batched=True)

    final_training_args = TrainingArguments(
        output_dir="./final_checkpoints",
        learning_rate=best_hyperparams["learning_rate"],
        per_device_train_batch_size=best_hyperparams["train_batch_size"],
        num_train_epochs=NUM_TRAIN_EPOCH,
        weight_decay=best_hyperparams["weight_decay"],
        save_strategy="epoch",
        save_steps=500,
        save_total_limit=2
    )

    final_trainer = Trainer(
        model=best_model,
        args=final_training_args,
        train_dataset=tokenized_train_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    final_trainer.train()
    final_trainer.save_model("./final_trained_model")

    final_trainer = Trainer(
        model=best_model,
        args=evaluation_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    test_results = final_trainer.evaluate()
    print("Final evaluation results on the test set: ", test_results)

train_test_all_models(3)

# THIS SHOULD WORK, BUT IT WILL TAKE TOO MUCH TIME, I SHOULD WAIT UNTIL I KNOW EXACTLY HOW I WANT TO TEST THE MODEL, BEFORE REALLY CHECKING THE BEST HYPER-PARAMS.




