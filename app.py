from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from huggingface_hub import login
import numpy as np
import torch

# Log in to Hugging Face Hub
login()  # You'll be prompted to enter your Hugging Face token

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load model and tokenizer
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Move model to GPU
model = model.to("cuda")

# Load Thai dataset (Wisesight Sentiment)
dataset = load_dataset("wisesight_sentiment")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch and move to GPU
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Reduced batch size
    per_device_eval_batch_size=8,   # Reduced batch size
    gradient_accumulation_steps=2,  # Gradient accumulation
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=True,
    hub_model_id="JonusNattapong/KaNomTom",
    fp16=True,  # Enable mixed precision for faster training
)

# Load evaluation metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save and push the model to Hugging Face Hub
model.push_to_hub("JonusNattapong/KaNomTom")
tokenizer.push_to_hub("JonusNattapong/KaNomTom")