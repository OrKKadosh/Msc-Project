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
def get_tokenizer(name):
  return AutoTokenizer.from_pretrained(name)

# Tokenize the datasets for each dataset
def tokenize_function(tokenizer, examples):
      return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=256)


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


model0 = {"tokenizer": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "model": AutoModelForSequenceClassification.from_pretrained(
    'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
          "name": "distilroberta-finetuned-financial-news-sentiment-analysis"}
model1 = {"tokenizer": "KernAI/stock-news-distilbert",
          "model": AutoModelForSequenceClassification.from_pretrained('KernAI/stock-news-distilbert', num_labels=3).to(
              device), "name": "stock-news-distilbert"}
model2 = {"tokenizer": "ProsusAI/finbert", "model": AutoModelForSequenceClassification.from_pretrained(
    'ProsusAI/finbert', num_labels=3).to(device),
          "name": "Finbert"}

models = [model0, model1, model2]

FPB = load_dataset("financial_phrasebank", 'sentences_75agree')['train']
train_FPB, test_FPB = FPB.train_test_split(test_size=0.3, seed=SEED).values()


evaluation_args = TrainingArguments(
    output_dir = "./results/checkpoints",
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    do_eval = True
)

LORA_FLAG = 0
test_dataset = test_FPB

for model in models:

    model_name = model["name"]
    tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
    tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    chosen_model = model["model"]

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

    results_dir = "./Evaluation_results/Baselines"
    os.makedirs(results_dir, exist_ok=True)
    results_file_name = model_name+ "diff_tokenizer" + ".txt"
    results_file_path = os.path.join(results_dir, results_file_name)

    with open(results_file_path, "w") as file:
        file.write(json.dumps(results_with_model, indent=4))

    print(f"Evaluation results for the model: {model_name} saved to baseline_evaluation_results.txt")