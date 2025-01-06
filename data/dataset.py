# Import required libraries for dataset handling (if needed)
from datasets import load_dataset

# Function to load dataset from Hugging Face
def load_dataset_from_huggingface(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

# You can expand this to load and prepare custom datasets if needed.
