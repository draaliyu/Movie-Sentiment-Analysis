import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np

def prepare_data():
    data = pd.read_csv('preprocessed_data.csv')
    X = data['review'].tolist()
    y = data['sentiment'].tolist()
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X)
    X = pad_sequences(tokenizer.texts_to_sequences(X), padding='post', maxlen=100)
    np.save('tokenizer.npy', tokenizer)
    np.save('prepared_data.npy', X)
    np.save('labels.npy', y)

    vocab_size = len(tokenizer.word_index) + 1
    np.save('vocab_size.npy', vocab_size)
    print("Data prepared and saved.")

def load_glove_embeddings(glove_file_path):
    tokenizer = np.load('tokenizer.npy', allow_pickle=True).item()
    embeddings_dictionary = {}
    with open(glove_file_path, encoding="utf8") as glove_file:
        for line in glove_file:
            records = line.split()
            if len(records) == 101:  # 1 word + 100 dimensions
                embeddings_dictionary[records[0]] = np.asarray(records[1:], dtype='float32')

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    np.save('embedding_matrix.npy', embedding_matrix)

if __name__ == "__main__":
    prepare_data()
    load_glove_embeddings('glove.6B.100d.txt')
