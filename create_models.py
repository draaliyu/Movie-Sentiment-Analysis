from keras.models import Sequential
from keras.layers import Flatten, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Embedding
import numpy as np


def create_model(model_type):
    vocab_size = np.load('vocab_size.npy').item()
    embedding_matrix = np.load('embedding_matrix.npy')

    model = Sequential()
    model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False))
    if model_type == 'dnn':
        model.add(Flatten())
    elif model_type == 'cnn':
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
    elif model_type == 'rnn':
        model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['acc'])
    return model


if __name__ == "__main__":
    models = {
        'dnn': create_model('dnn'),
        'cnn': create_model('cnn'),
        'rnn': create_model('rnn')
    }

    for model_type, model in models.items():
        model.save(f"{model_type}_model.h5")
        print(f"{model_type.upper()} model saved.")
