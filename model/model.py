from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_name="airesearch/wangchanberta-base-att-spm-uncased", num_labels=4):
    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return model, tokenizer
