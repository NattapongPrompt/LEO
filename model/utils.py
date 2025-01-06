from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer, util

# Initialize pipelines for different tasks
qa_pipeline = pipeline("question-answering")
zero_shot_classifier = pipeline("zero-shot-classification")
generator = pipeline("text-generation")
summarizer = pipeline("summarization")
mask_filler = pipeline("fill-mask")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function for Text Classification
def classify_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    return prediction.item()

# Token Classification (NER)
def extract_entities(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_tokens = torch.argmax(outputs.logits, dim=-1)
    entities = tokenizer.convert_ids_to_tokens(predicted_tokens[0])
    return entities

# Question Answering
def answer_question(question, context):
    return qa_pipeline({'context': context, 'question': question})

# Zero-Shot Classification
def zero_shot_classify(text, candidate_labels):
    return zero_shot_classifier(text, candidate_labels)

# Text Generation
def generate_text(prompt):
    return generator(prompt, max_length=50, num_return_sequences=1)

# Summarization
def summarize_text(text):
    return summarizer(text, max_length=150, min_length=30, do_sample=False)

# Sentence Similarity
def calculate_similarity(sent1, sent2):
    emb1 = sentence_model.encode(sent1, convert_to_tensor=True)
    emb2 = sentence_model.encode(sent2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2)
    return similarity.item()

# Fill-Mask
def fill_mask(text):
    return mask_filler(text)
