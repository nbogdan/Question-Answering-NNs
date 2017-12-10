import pickle
import re

import keras
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

MAX_SEQUENCE_LENGTH_Q = 100
MAX_SEQUENCE_LENGTH_A = 20
MAX_SEQUENCE_LENGTH_C = 500
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
LEARNING_PREPROC = False

def checkModelForFolder(modelName, folderName, testData, weightsFile):
    context_data, question_data, answer_data, y_test, text = testData
    json_file = open(folderName + 'structures/' + modelName, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(folderName + 'structures/' + weightsFile)

    opt = keras.optimizers.Nadam()
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    result = loaded_model.predict({'context': context_data, 'question': question_data, 'answer': answer_data}, batch_size=300)
    real_c = [np.argmax(x) for x in y_test]

    correct = 0
    total = 0
    for i in range(int(len(real_c) / 4)):
        local_res = result[i*4:i*4+4]
        real_result = np.argmax(real_c[i*4:i*4+4])
        aux = np.argmax([x[1] for x in local_res])
        if aux == real_result:
            correct+=1
        total+=1
    print(correct)
    print(total)
    print(correct / total)

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

    return [data_c, data_q, data_a, labels, data_train]

def checkModel(modelName, folderName, weightsFile):
    checkModelForFolder(modelName, folderName, loadTestData(folderName), weightsFile)

if __name__ == '__main__':
    # preprocessData("data_test_extra/")
    model = "cnn"
    print('Testing model %', model)
    checkModel(model + "-model1.json", "data_test_small/", 'cnn1-final-01-0.52.hdf5')
