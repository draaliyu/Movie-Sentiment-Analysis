import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os


def train_and_evaluate(model_type, X_train, y_train, X_test, y_test, batch_size=16, epochs=20):
    model = load_model(f"{model_type}_model.h5")
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)
    score = model.evaluate(X_test, y_test, verbose=1)
    trained_model_path = f"{model_type}_model_trained.h5"
    model.save(trained_model_path)
    print(f"Trained {model_type.upper()} model saved to {trained_model_path}")
    return history.history, score


if __name__ == "__main__":
    # Load data
    X = np.load('prepared_data.npy')
    y = np.load('labels.npy')
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=70)

    # Dictionaries to store histories and scores
    histories, scores = {}, {}

    # Train and evaluate each model type
    for model_type in ['dnn', 'cnn', 'rnn']:
        batch_size = 8 if model_type == 'dnn' else 128
        history, score = train_and_evaluate(model_type, X_train, y_train, X_test, y_test, batch_size=batch_size)
        histories[model_type] = history
        scores[model_type] = score
        print(f"\n{model_type.upper()} MODEL")
        print(f"Test loss: {score[0]}")
        print(f"Test Accuracy: {score[1]}")

    # Save histories and scores
    np.save('histories.npy', histories)
    np.save('scores.npy', scores)
    print("Training histories and scores saved.")
