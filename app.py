from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
import numpy as np
import torch
import evaluate

# Log in to Hugging Face Hub
login()  # Prompt to enter your Hugging Face token

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load model and tokenizer
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Move model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load the Wisesight Sentiment dataset
dataset = load_dataset("wisesight_sentiment")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train_dataset = tokenized_datasets["train"].shuffle(seed=42)
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)

# Load evaluation metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {**accuracy, **f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=True,
    hub_model_id="JonusNattapong/KaNomTom",
    fp16=torch.cuda.is_available(),
    lr_scheduler_type="linear",  # Learning rate scheduler
    load_best_model_at_end=True,  # Save best model
    metric_for_best_model="accuracy",
    save_total_limit=2,  # Limit the number of checkpoints
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

# Push model and tokenizer to Hugging Face Hub
trainer.push_to_hub()
tokenizer.push_to_hub("JonusNattapong/KaNomTom")
