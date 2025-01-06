from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import torch
import numpy as np
import evaluate

def train_and_evaluate(model, tokenizer, output_dir="./results", hub_model_id="JonusNattapong/KaNomTom"):
    # Load dataset
    dataset = load_dataset("pythainlp/wisesight_sentiment")
    def tokenize_function(examples):
        return tokenizer(examples["texts"], padding="max_length", truncation=True, max_length=512)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["texts"])
    tokenized_datasets = tokenized_datasets.rename_column("category", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Split dataset into train and eval sets
    split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        push_to_hub=True,
        hub_model_id=hub_model_id,
        load_best_model_at_end=True
    )

    # Define evaluation metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    # Train and evaluate the model
    trainer.train()
    trainer.evaluate()

    # Push model and tokenizer to Hugging Face Hub
    model.push_to_hub(hub_model_id)
    tokenizer.push_to_hub(hub_model_id)
