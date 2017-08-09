import numpy as np

import keras
from keras.models import model_from_json
import pandas as pd
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import random
from bs4 import BeautifulSoup
from siamese.data_helpers import get_lemmas
import re
from nltk.stem import WordNetLemmatizer

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
    data_test = pd.read_csv(folder + 'data/classic_test_datum.txt', sep='\t')
    test_texts_q = []
    test_texts_a = []
    test_texts_c = []

    for idx in range(data_test.question.shape[0]):
        text = BeautifulSoup(data_test.question[idx], "html.parser")
        test_texts_q.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
        text = BeautifulSoup(data_test.answer[idx], "html.parser")
        test_texts_a.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
        text = BeautifulSoup(str(data_test.context[idx]), "html.parser")
        test_texts_c.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])

    lemmatizer = WordNetLemmatizer()
    test_texts_c3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in test_texts_c]
    test_texts_q3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in test_texts_q]
    test_texts_a3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in test_texts_a]

    with open(folder + 'c_test_lemmas_c', 'wb') as fp:
        pickle.dump(test_texts_c3, fp)
    with open(folder + 'c_test_lemmas_q', 'wb') as fp:
        pickle.dump(test_texts_q3, fp)
    with open(folder + 'c_test_lemmas_a', 'wb') as fp:
        pickle.dump(test_texts_a3, fp)
    print('Saved lemmas')


def checkModelForFolder(modelName, folderName, testData):
    context_data, question_data, answer_data, y_test = testData
    json_file = open(folderName + 'structures/' + modelName, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(folderName + 'structures/nn1-final-03-0.56.hdf5')

    opt = keras.optimizers.Nadam()
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    result2 = loaded_model.predict({'context': context_data, 'question': question_data, 'answer': answer_data}, batch_size=300)
    real_c = [np.argmax(x) for x in y_test]

    correct = 0
    total = 0
    for i in range(int(len(real_c) / 4)):
        results = result2[i*4:i*4+4]
        real_result = np.argmax(real_c[i*4:i*4+4])
        aux = np.argmax([x[1] for x in results])
        print(str(aux) + ' ' + str(results))
        if aux == real_result:
            correct+=1
        total+=1
    print(correct)
    print(total)

def loadTestData(folderName):
    data_train = pd.read_csv(folderName + 'data/classic_test_datum.txt', sep='\t', error_bad_lines=False)
    labels = []
    for idx in range(data_train.question.shape[0]):
        labels.append(data_train.value[idx])
    texts_c3 = pickle.load(open(folderName + 'c_test_lemmas_c', 'rb'))
    texts_q3 = pickle.load(open(folderName + 'c_test_lemmas_q', 'rb'))
    texts_a3 = pickle.load(open(folderName + 'c_test_lemmas_a', 'rb'))

    tokenizer = pickle.load(open(folderName + 'structures/tokenizer', 'rb'))
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

    embedding_matrix = pickle.load(open(folderName + 'structures/embedding_matrix', 'rb'))

    for i in range(int(len(data_c) / 4)):
        x = random.randint(0,3)
        aux = list([data_c[i * 4 + x], data_q[i * 4 + x], data_a[i * 4 + x], labels[i * 4 + x]])
        data_c[i * 4 + x], data_q[i * 4 + x], data_a[i * 4 + x], labels[i * 4 + x] = [data_c[i * 4], data_q[i * 4], data_a[i * 4], labels[i * 4]]
        data_c[i * 4], data_q[i * 4], data_a[i * 4], labels[i * 4] = aux
    return [data_c, data_q, data_a, labels]

def red(modelName, folderName):
    data = loadTestData(folderName)

    checkModelForFolder(modelName, folderName, data)

if __name__ == '__main__':
    # preprocessData("data_test_extra/")
    red("cnn-model1.json", "data_test_extra/")
