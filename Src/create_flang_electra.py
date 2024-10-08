import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, \
    Trainer

# Load the model and tokenizer
model_name = "SALT-NLP/FLANG-ELECTRA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels for sentiment analysis (negative, neutral, positive)


df = pd.read_csv('/cs_storage/orkados/FinGPT_cleaned_dataset.csv')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert the Pandas DataFrames to Hugging Face `datasets`
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Step 1: Map the sentiment labels to integers
def map_labels(examples):
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    examples['labels'] = label_map[examples['output']]  # Map 'output' directly using Python's dictionary
    return examples

# Apply this mapping to both train and test datasets
train_dataset = train_dataset.map(map_labels)
test_dataset = test_dataset.map(map_labels)

# Step 2: Preprocess the dataset (tokenize the text and include labels)
def preprocess_function(examples):
    # Tokenize the input text
    inputs = tokenizer(examples['input'], truncation=True, padding=True)
    # Include the labels in the tokenized output
    inputs['labels'] = examples['labels']
    return inputs

# Tokenize the dataset
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",           # Directory to save checkpoints
    evaluation_strategy="epoch",      # Evaluate every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Use a data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define a function to compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Pass it to the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # Add the accuracy computation
)

# Train the model
trainer.train()

# Evaluate the model and check accuracy
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']}")
