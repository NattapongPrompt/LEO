from model.model import load_model
from model.trainer import train_and_evaluate

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()

    # Train and evaluate the model
    train_and_evaluate(model, tokenizer)

if __name__ == "__main__":
    main()
