import os
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
from datasets import load_metric, load_dataset, concatenate_datasets
import numpy as np
import json
from datetime import datetime  # Import datetime module


SEED = 1694
np.random.seed(SEED)
torch.manual_seed(SEED)

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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


def compute_metrics(eval_pred, model_name):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  # needed while using the FINBERT & base_stock-news-distilbert, since its labels are not matching
  if 'Finbert' in model_name:
      id2label = {0: 2, 1: 0, 2: 1}
      mapped_predictions = [id2label[pred] for pred in predictions]
  elif 'stock-news-distilbert' in model_name:
      id2label = {0: 1, 1: 0, 2: 2}
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


base_model0 = {"tokenizer": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "model": AutoModelForSequenceClassification.from_pretrained(
    'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', num_labels=3).to(device),
          "name": f"base_distilroberta-finetuned-financial-news-sentiment-analysis{now}"}
base_model1 = {"tokenizer": "KernAI/stock-news-distilbert",
          "model": AutoModelForSequenceClassification.from_pretrained('KernAI/stock-news-distilbert', num_labels=3).to(
              device), "name": f"base_stock-news-distilbert{now}"}
base_model2 = {"tokenizer": "ProsusAI/finbert", "model": AutoModelForSequenceClassification.from_pretrained(
    'ProsusAI/finbert', num_labels=3).to(device),
          "name": f"base_Finbert{now}"}

# base_models = [base_model1]
base_models = [base_model0, base_model1, base_model2]

PT_model0 = {"save_directory": "./Saved_models/pre_trained/distilroberta-finetuned-financial-news-sentiment-analysis",
                     "name": "Basic_PT_distilroberta-finetuned-financial-news-sentiment-analysis"}
PT_model1 = {"save_directory" : "./Saved_models/pre_trained/stock-news-distilbert",
                    "name": "Basic_PT_stock-news-distilbert"}
PT_model2 = {"save_directory" : "./Saved_models/pre_trained/Finbert",
                    "name": "Basic_PT_Finbert"}
PT_models = [PT_model0, PT_model1, PT_model2]


model0_PT_withRID = {"save_directory": "./Saved_models/pre_trained/only RIDdistilroberta-finetuned-financial-news-sentiment-analysis",
                     "name": "Basic+RID_PT_distilroberta-finetuned-financial-news-sentiment-analysis"}
model1_PT_withRID ={"save_directory" : "./Saved_models/pre_trained/only RIDstock-news-distilbert",
                    "name": "Basic+RID_PT_stock-news-distilbert"}
PT_RID_models = [model0_PT_withRID, model1_PT_withRID]

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

for model in base_models:
    model_name = model["name"]
    tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
    tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    chosen_model = model["model"]
    if 'Finbert' in model_name:     id2label = {0:2, 1:0, 2:1}  # Retrieve id2label from model config
    elif 'stock-news-distilbert' in model_name: id2label = {0: 1, 1: 0, 2: 2}
    else : id2label = None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

    trainer = Trainer(
        model=chosen_model,
        args=evaluation_args,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred : compute_metrics(eval_pred, model_name),
    )

    # Get predictions
    predictions = trainer.predict(tokenized_test_dataset)

    # Labels: 0 is negative, 1 is neutral, 2 is positive.

    # Extract logits and labels
    logits = predictions.predictions
    true_labels = predictions.label_ids

    # Get predicted labels
    predicted_labels = np.argmax(logits, axis=-1)

    # Collect results for each sample
    detailed_results = []
    count_predicted_label = {0: 0, 1: 0, 2: 0}
    count_real_label = {0: 0, 1: 0, 2: 0}

    count_mistakes = {"true 0 to predicted 1": 0, "true 0 to predicted 2":0, "true 1 to predicted 0":0, "true 1 to predicted 2": 0, "true 2 to predicted 0":0, "true 2 to predicted 1": 0}


    # Counts the labels of the test_set samples and the predicted labels
    for idx, sample in enumerate(test_dataset):
        sentence = sample['sentence']

        true_label = int(true_labels[idx])
        if id2label: predicted_label = int(id2label[predicted_labels[idx]])
        else: predicted_label = int(predicted_labels[idx])

        if true_label != predicted_label:
            if true_label == 0 and predicted_label == 1: count_mistakes["true 0 to predicted 1"] += 1
            elif true_label == 0 and predicted_label == 2: count_mistakes["true 0 to predicted 2"] += 1
            elif true_label == 1 and predicted_label == 0: count_mistakes["true 1 to predicted 0"] += 1
            elif true_label == 1 and predicted_label == 2: count_mistakes["true 1 to predicted 2"] += 1
            elif true_label == 2 and predicted_label == 0: count_mistakes["true 2 to predicted 0"] += 1
            elif true_label == 2 and predicted_label == 1: count_mistakes["true 2 to predicted 1"] += 1

        count_real_label[true_label] += 1
        count_predicted_label[predicted_label] += 1

        # Append the result for this sample to detailed_results
        detailed_results.append({
            "sample": sentence,
            "true_label": int(true_label),  # Use id2label to get the correct label
            "predicted_label": int(predicted_label)  # Use id2label for predicted label
        })

    # Collect all evaluation results including the detailed sample predictions
    results_with_model = {
        "model_name": model_name,
        "overall_results": trainer.evaluate(),  # Overall evaluation metrics
        "count_real_label" : count_real_label,
        "count_predicted_label" : count_predicted_label,
        "count_mistakes" : count_mistakes,
        "detailed_results": detailed_results  # Add detailed sample-level results
    }

    # Create results directory and save to file
    results_dir = "./Evaluation_results/Baselines"
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, model_name + ".json")

    # Save the results as JSON
    with open(results_file_path, "w") as file:
        json.dump(results_with_model, file, indent=4)

    print(f"Evaluation results for the model: {model_name} saved to {results_file_path}")

print("Ended evaluation of base models")


# for model in base_models:
#
#     model_name = model["name"]
#     tokenizer = AutoTokenizer.from_pretrained(model["tokenizer"])
#     tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)
#
#     chosen_model = model["model"]
#
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
#
#     trainer = Trainer(
#         model=chosen_model,
#         args=evaluation_args,
#         eval_dataset=tokenized_test_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )
#
#     results_with_model = {
#         "model_name" : model_name,
#         "results" : trainer.evaluate()
#     }
#
#     results_dir = "./Evaluation_results/Baselines"
#     os.makedirs(results_dir, exist_ok=True)
#     results_file_path = os.path.join(results_dir, model_name)
#
#     with open(results_file_path, "w") as file:
#         file.write(json.dumps(results_with_model, indent=4))
#
#     print(f"Evaluation results for the model: {model_name} saved to ./Evaluation_results/Baselines")
#
# print("ended evaluation of base models")
#
# for model in PT_models:
#
#     model_name = model["name"]
#     chosen_model = AutoModelForSequenceClassification.from_pretrained(model["save_directory"])
#     tokenizer = AutoTokenizer.from_pretrained(model["save_directory"])
#     tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)
#
#     # chosen_model = model["model"]
#
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
#
#     trainer = Trainer(
#         model=chosen_model,
#         args=evaluation_args,
#         eval_dataset=tokenized_test_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )
#
#     results_with_model = {
#         "model_name" : model_name,
#         "results" : trainer.evaluate()
#     }
#
#     results_dir = "./Evaluation_results/Baselines"
#     os.makedirs(results_dir, exist_ok=True)
#     results_file_name = model_name +".txt"
#     results_file_path = os.path.join(results_dir, results_file_name)
#
#     with open(results_file_path, "w") as file:
#         file.write(json.dumps(results_with_model, indent=4))
#
#     print(f"Evaluation results for the model: {model_name} saved to ./Evaluation_results/Baselines")
#
# print("ended evaluation of PT_models")

# for model in PT_RID_models:
#
#     model_name = model["name"]
#     chosen_model = AutoModelForSequenceClassification.from_pretrained(model["save_directory"])
#     tokenizer = AutoTokenizer.from_pretrained(model["save_directory"])
#     tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)
#
#     # chosen_model = model["model"]
#
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
#
#     trainer = Trainer(
#         model=chosen_model,
#         args=evaluation_args,
#         eval_dataset=tokenized_test_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )
#
#     results_with_model = {
#         "model_name" : model_name,
#         "results" : trainer.evaluate()
#     }
#
#     results_dir = "./Evaluation_results/Baselines"
#     os.makedirs(results_dir, exist_ok=True)
#     results_file_name = model_name + ".txt"
#     results_file_path = os.path.join(results_dir, results_file_name)
#
#     with open(results_file_path, "w") as file:
#         file.write(json.dumps(results_with_model, indent=4))
#
#     print(f"Evaluation results for the model: {model_name} saved to ./Evaluation_results/Baselines")
#
# print("ended evaluation of PT_RID_models")
# print("ended evaluation")
