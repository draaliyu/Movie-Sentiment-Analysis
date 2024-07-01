# Sentiment Analysis Project

This project performs sentiment analysis on IMDB movie reviews using deep learning models. The workflow is divided into several scripts, each responsible for a specific part of the process, from data loading and preprocessing to model creation, training, evaluation, and visualization of results.

## Project Structure
- ├── load_data.py
- ├── preprocess_data.py
- ├── prepare_data.py
- ├── create_models.py
- ├── train_evaluate.py
- ├── plots.py
- └── main.py


### Scripts

- **load_data.py**: Loads and inspects the IMDB dataset.
- **preprocess_data.py**: Preprocesses the text data by removing HTML tags, punctuation, and numbers.
- **prepare_data.py**: Prepares the data for model input, including tokenization and padding. It also loads GloVe embeddings.
- **create_models.py**: Creates deep learning models (DNN, CNN, RNN) and saves them to disk.
- **train_evaluate.py**: Trains and evaluates the models, saving the trained models and training histories.
- **plots.py**: Plots the training histories for accuracy and loss.
- **main.py**: Runs all the above scripts in the correct sequence.

## Requirements

- Python 3.x
- pandas
- numpy
- nltk
- matplotlib
- keras
- sklearn

To install the required packages, you can use:

```bash
pip install pandas numpy nltk matplotlib keras scikit-learn
```

## Data

The dataset used is the IMDB movie reviews dataset. Ensure the file IMDB Dataset.csv is in the project directory.

## Pre-trained Embeddings
This project uses GloVe embeddings. Download the glove.6B.100d.txt file and place it in the project directory.

## Running the Project
To execute the entire workflow, simply run the main.py script:
```bash
python3 main.py
```
This will sequentially execute all the scripts, performing the following steps:

1. Load and inspect the data.
2.  Preprocess the text data.
3. Prepare the data for model input.
4. Create and save the models.
5. Train and evaluate the models, saving the training histories.
6. Plot the training histories for accuracy and loss.

## Troubleshooting
- Ensure all required files (IMDB Dataset.csv and glove.6B.100d.txt) are in the project directory.
- Check that all Python packages are installed.
- Ensure that you have read and write permissions in the project directory.




### This `README.md` provides a comprehensive overview of the project, including its structure, requirements, and instructions for running the entire workflow. Adjust paths and filenames as necessary to fit your specific setup.
