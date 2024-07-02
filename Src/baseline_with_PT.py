import os

import torch

from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
from datasets import load_metric, load_dataset, concatenate_datasets
import numpy as np
import json

SEED = 1694
np.random.seed(SEED)
torch.manual_seed(SEED)

# BASE-LINE: no pre-train and no fine-tuning
# Model name: distilroberta-finetuned-financial-news-sentiment-analysis
# DataSet: 75agree + allagree of PFB


# Make sure a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Baseline is Using device: {device}")
# Load the metrics
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")

# get the initialized tokenizer
# def get_tokenizer(name):
#   return AutoTokenizer.from_pretrained(name)

# Tokenize the datasets for each dataset
def tokenize_function(tokenizer, examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=256)


# Encode labels as integers according to: 0-negative, 1-neutral, 2-positive
# label_dict0 = {0: 1, 1: 2, 2: 0} #financial-tweets-sentiment
# label_dict1 = {0: 1, 1: 2, 2: 0} #synthetic-financial-tweets-sentiment
# label_dict3 = {0: 0, 1: 1, 2: 2} #financial_phrasebank
# label_dict4 = {0: 0, 1: 2, 2: 1} #twitter-financial-news-sentiment

# def encode_labels(example, ds):
#   if ds == 0: #financial-tweets-sentiment
#     example['label'] = label_dict0[example['sentiment']]
#   elif ds == 1: #synthetic-financial-tweets-sentiment
#     example['label'] = label_dict1[example['sentiment']]
#   elif ds == 2: #fiqa-sentiment-classification
#     if example['score'] <= -0.4: example['label'] = 0
#     elif example['score'] <= 0.5: example['label'] = 1
#     else:
#       example['label'] = 2
#   elif ds == 3: #financial_phrasebank
#     example['label'] = label_dict3[example['label']]
#   elif ds == 4: #twitter-financial-news-sentiment
#     example['label'] = label_dict4[example['label']]
#   return example

def compute_metrics(eval_pred):
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


model0 = {"save_directory": "./Saved_models/pre_trained/distilroberta-finetuned-financial-news-sentiment-analysis",
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}
model1 = {"save_directory": "./Saved_models/pre_trained/stock-news-distilbert",
          "name": "stock-news-distilbert"}
model2 = {"save_directory": "./Saved_models/pre_trained/Finbert",
          "name": "Finbert"}

models = [model0, model1, model2]

# LORA:
# lora_rank = [4, 8, 16]
# lora_alpha = lora_rank * 2
# # lora_alphas = lora_rank * [1, 1.5, 2]
# lora_dropout = [0.0, 0.05, 0.1, 0.2]
# idx_lora = 0
# lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_rank[idx_lora], lora_alpha=lora_alpha[idx_lora], lora_dropout=lora_dropout[idx_lora])
# evaluation_results = {}

FPB = load_dataset("financial_phrasebank", 'sentences_75agree')['train']
train_FPB, test_FPB = FPB.train_test_split(test_size=0.3, seed=SEED).values()



NUM_TRAIN_EPOCH = 3

evaluation_args = TrainingArguments(
    output_dir = "./results/checkpoints",
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    do_eval = True
)

LORA_FLAG = 0
test_dataset = test_FPB
NUM_DATASETS = 4

for model in models:

    model_name = model['name']
    save_directory = model["save_directory"]
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    chosen_model = AutoModelForSequenceClassification.from_pretrained(save_directory)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

    trainer = Trainer(
        model=chosen_model,
        args=evaluation_args,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    results_with_model = {
        "model_name" : model_name,
        "results" : trainer.evaluate()
    }

    results_dir = "./Evaluation_results/Baselines/with_PT"
    os.makedirs(results_dir, exist_ok=True)
    results_file_name = model_name + ".txt"
    results_file_path = os.path.join(results_dir, results_file_name)

    with open(results_file_path, "w") as file:
        file.write(json.dumps(results_with_model, indent=4))

    print(f"Evaluation results for the model: {model_name} saved to baseline_evaluation_results.txt")