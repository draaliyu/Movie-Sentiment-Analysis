import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


def preprocess_text(text):
    import re
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove punctuations and numbers
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # Single character removal
    text = re.sub(r'\s+', ' ', text)  # Removing multiple spaces
    return text.strip()


def load_tokenizer():
    return np.load('tokenizer.npy', allow_pickle=True).item()


def load_model_and_predict(model_type, review):
    # Load the trained model
    model = load_model(f"{model_type}_model_trained.h5")

    # Load the tokenizer
    tokenizer = load_tokenizer()

    # Preprocess the review
    review_processed = preprocess_text(review)
    review_sequence = tokenizer.texts_to_sequences([review_processed])
    review_padded = pad_sequences(review_sequence, padding='post', maxlen=100)

    # Predict sentiment
    prediction = model.predict(review_padded)
    sentiment = 'positive' if prediction > 0.5 else 'negative'

    return sentiment, prediction[0][0]


if __name__ == "__main__":
    # User input for review and model type
    review = input("Enter the review text: ")
    model_type = input("Enter the model type (dnn, cnn, rnn): ").lower()

    if model_type not in ['dnn', 'cnn', 'rnn']:
        print("Invalid model type. Please enter 'dnn', 'cnn', or 'rnn'.")
    else:
        sentiment, probability = load_model_and_predict(model_type, review)
        print(f"\nPrediction by {model_type.upper()} model:")
        print(f"Sentiment: {sentiment}")
        print(f"Probability: {probability:.4f}")
