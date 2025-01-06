from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset(model_name="airesearch/wangchanberta-base-att-spm-uncased"):
    # Load the dataset
    dataset = load_dataset("pythainlp/wisesight_sentiment")

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["texts"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["texts"])
    tokenized_datasets = tokenized_datasets.rename_column("category", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_datasets
