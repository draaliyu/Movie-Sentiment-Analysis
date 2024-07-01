import matplotlib.pyplot as plt
import numpy as np

def plot_histories(histories):
    for model_name, history in histories.items():
        plt.plot(history['acc'], label=f'{model_name.upper()} Train')
        plt.plot(history['val_acc'], label=f'{model_name.upper()} Valid')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of epoch')
    plt.legend(loc='best')
    plt.show()

    for model_name, history in histories.items():
        plt.plot(history['loss'], label=f'{model_name.upper()} Train')
        plt.plot(history['val_loss'], label=f'{model_name.upper()} Valid')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Number of epoch')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    histories = np.load('histories.npy', allow_pickle=True).item()
    plot_histories(histories)
