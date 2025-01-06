import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from model.model import load_model, load_tokenizer
from transformers import EarlyStoppingCallback

class MultiTaskTrainer:
    def __init__(self, task_type, model_name, dataset, output_dir):
        self.task_type = task_type
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir

        self.model = load_model(task_type, model_name)
        self.tokenizer = load_tokenizer(model_name)

        # Tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)

        # Set format for PyTorch
        self.dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Split the dataset into train and validation sets
        split_dataset = self.dataset["train"].train_test_split(test_size=0.1, seed=42)
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
        )

        early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping],
        )

        trainer.train()
        trainer.evaluate()

        # Save model
        self.model.save_pretrained(self.output_dir)

