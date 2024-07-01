import pandas as pd
import re

def preprocess(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove punctuations and numbers
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # Single character removal
    text = re.sub(r'\s+', ' ', text)  # Removing multiple spaces
    return text.strip()

def preprocess_reviews(file_path):
    movie_reviews = pd.read_csv(file_path)
    X = [preprocess(review) for review in movie_reviews['review']]
    y = [1 if sentiment == "positive" else 0 for sentiment in movie_reviews['sentiment']]
    pd.DataFrame({'review': X, 'sentiment': y}).to_csv('preprocessed_data.csv', index=False)

if __name__ == "__main__":
    preprocess_reviews('IMDB Dataset.csv')
