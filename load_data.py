import pandas as pd

def load_and_inspect_data(file_path):
    movie_reviews = pd.read_csv(file_path)
    print("\nIs the dataset having any null entries?", movie_reviews.isnull().values.any())
    print("\nDataset dimensions", movie_reviews.shape)
    print("\nFirst 5 entries in dataset", movie_reviews.head())
    print("\nNumber of positive and negative reviews in dataset", movie_reviews['sentiment'].value_counts())
    movie_reviews.to_csv('loaded_data.csv', index=False)

if __name__ == "__main__":
    load_and_inspect_data("IMDB Dataset.csv")
