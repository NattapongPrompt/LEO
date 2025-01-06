from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import torch
import evaluate
from torch.optim import AdamW
from transformers import get_scheduler


# Log in to Hugging Face Hub
login()


# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")


# Load model and tokenizer
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)


# Move model to GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# Load the dataset
dataset = load_dataset("pythainlp/wisesight_sentiment")

# Check dataset structure
print(dataset["train"].column_names)  # Verify the correct column names
print(dataset["train"][0])  # Inspect a sample entry


# Tokenize dataset using the correct column name
def tokenize_function(examples):
    return tokenizer(examples["texts"], padding="max_length", truncation=True, max_length=512)


# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["texts"])  # Remove "texts"
tokenized_datasets = tokenized_datasets.rename_column("category", "labels")  # Rename "category" to "labels"

# Set format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Split the train dataset into train and validation sets
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,   # Adjust based on GPU memory
    gradient_accumulation_steps=1,  # Adjust based on GPU memory
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=True,
    hub_model_id="JonusNattapong/KaNomTom",
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,  # Load the best model at the end of training
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_total_limit=2,  # Keep only the last 2 checkpoints to avoid large storage usage
)

# Define evaluation metric
metric = evaluate.load("accuracy")  # Requires scikit-learn

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Early stopping callback to avoid overfitting
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)


# Train the model
trainer.train()


# Evaluate the model
trainer.evaluate()


# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub("JonusNattapong/KaNomTom")
tokenizer.push_to_hub("JonusNattapong/KaNomTom")
