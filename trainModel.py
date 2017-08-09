# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd


from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import re, string
from nltk.stem import WordNetLemmatizer
import pickle

from models.simpleCNN import SimpleCNNModel
from models.simpleLSTM import SimpleLSTMModel
from siamese.data_helpers import get_lemmas

MAX_SEQUENCE_LENGTH_Q = 100
MAX_SEQUENCE_LENGTH_A = 20
MAX_SEQUENCE_LENGTH_C = 500
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
LEARNING_PREPROC = False

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def preprocessData(folder):
    data_train = pd.read_csv(folder + 'data/train_datum.txt', sep='\t', error_bad_lines=False)
    data_test = pd.read_csv(folder + 'data/classic_test_datum.txt', sep='\t')
    texts_q = []
    texts_a = []
    texts_c = []
    test_texts_q = []
    test_texts_a = []
    test_texts_c = []

    for idx in range(data_train.question.shape[0]):
        text = BeautifulSoup(data_train.question[idx], "html.parser")
        texts_q.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
        text = BeautifulSoup(data_train.answer[idx], "html.parser")
        texts_a.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
        text = BeautifulSoup(str(data_train.context[idx]), "html.parser")
        texts_c.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    for idx in range(data_test.question.shape[0]):
        text = BeautifulSoup(data_train.question[idx], "html.parser")
        test_texts_q.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
        text = BeautifulSoup(data_train.answer[idx], "html.parser")
        test_texts_a.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
        text = BeautifulSoup(str(data_train.context[idx]), "html.parser")
        test_texts_c.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])

    lemmatizer = WordNetLemmatizer()
    texts_c3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_c]
    texts_q3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_q]
    texts_a3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_a]

    test_texts_c3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_c]
    test_texts_q3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_q]
    test_texts_a3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_a]

    with open(folder + 'train_lemmas_c', 'wb') as fp:
        pickle.dump(texts_c3, fp)
    with open(folder + 'train_lemmas_q', 'wb') as fp:
        pickle.dump(texts_q3, fp)
    with open(folder + 'train_lemmas_a', 'wb') as fp:
        pickle.dump(texts_a3, fp)
    with open(folder + 'test_lemmas_c', 'wb') as fp:
        pickle.dump(test_texts_c3, fp)
    with open(folder + 'test_lemmas_q', 'wb') as fp:
        pickle.dump(test_texts_q3, fp)
    with open(folder + 'test_lemmas_a', 'wb') as fp:
        pickle.dump(test_texts_a3, fp)
    print('Saved lemmas')

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_q3 + texts_c3 + texts_a3 + test_texts_a3 + test_texts_c3 + test_texts_q3)
    print('Saving tokenizer')
    with open(folder + 'structures/tokenizer', 'wb') as fp:
        pickle.dump(tokenizer, fp)

    embeddings_index = {}
    f = open('glove.6B.300d.txt', 'r', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    word_index = tokenizer.word_index
    if True or LEARNING_PREPROC:
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        print('Saving embedding matrix')
        with open(folder + 'structures/embedding_matrix', 'wb') as fp:
            pickle.dump(embedding_matrix, fp)

def loadAndPrepareDataTrain(folder):
    data_train = pd.read_csv(folder + 'data/train_datum.txt', sep='\t', error_bad_lines=False)
    labels = []
    for idx in range(data_train.question.shape[0]):
        labels.append(data_train.value[idx])
    texts_c3 = pickle.load(open(folder + 'train_lemmas_c', 'rb'))
    texts_q3 = pickle.load(open(folder + 'train_lemmas_q', 'rb'))
    texts_a3 = pickle.load(open(folder + 'train_lemmas_a', 'rb'))

    tokenizer = pickle.load(open(folder + 'structures/tokenizer', 'rb'))
    sequences_q = tokenizer.texts_to_sequences(texts_q3)
    sequences_a = tokenizer.texts_to_sequences(texts_a3)
    sequences_c = tokenizer.texts_to_sequences(texts_c3)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data_q = pad_sequences(sequences_q, maxlen=MAX_SEQUENCE_LENGTH_Q)
    data_a = pad_sequences(sequences_a, maxlen=MAX_SEQUENCE_LENGTH_A)
    data_c = pad_sequences(sequences_c, maxlen=MAX_SEQUENCE_LENGTH_C)

    labels = to_categorical(np.asarray(labels))
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data_q.shape[0])
    np.random.shuffle(indices)
    data_q = data_q[indices]
    data_a = data_a[indices]
    data_c = data_c[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data_q.shape[0])

    question_data = data_q[:-nb_validation_samples]
    answer_data = data_a[:-nb_validation_samples]
    context_data = data_c[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]

    question_data_v = data_q[-nb_validation_samples:]
    answer_data_v = data_a[-nb_validation_samples:]
    context_data_v = data_c[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Traing and validation set number of positive and negative reviews')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    embedding_matrix = pickle.load(open(folder + 'structures/embedding_matrix', 'rb'))

    return {
        "train": [context_data, question_data, answer_data, y_train],
        "val": [context_data_v, question_data_v, answer_data_v, y_val],
        "word_index": word_index,
        "embedding_matrix": embedding_matrix
    }

def trainModelOnFolder(modelName, folderName):
    data = loadAndPrepareDataTrain(folderName)

    if modelName == "simpleCNN":
        model = SimpleCNNModel(data['word_index'], data['embedding_matrix'])
        model.train(data['train'], data['val'], folderName)
    if modelName == "simpleLSTM":
        model = SimpleLSTMModel(data['word_index'], data['embedding_matrix'])
        model.train(data['train'], data['val'], folderName)

if __name__ == '__main__':
    trainModelOnFolder("simpleLSTM", "data_test_extra/")