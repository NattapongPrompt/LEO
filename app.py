from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
import torch

# Log in to Hugging Face Hub
login()

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load model and tokenizer (T5 for Text Generation)
model_name = "t5-small"  # Or use another model that supports text generation
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load the dataset (choose a dataset that fits your task, e.g., DailyDialog for conversations)
dataset = load_dataset("daily_dialog")  # Example dataset for dialogue

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['dialog'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Split the dataset into train and validation
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Push the model and tokenizer to Hugging Face Hub
model.push_to_hub("JonusNattapong/KaNomTom")
tokenizer.push_to_hub("JonusNattapong/KaNomTom")
