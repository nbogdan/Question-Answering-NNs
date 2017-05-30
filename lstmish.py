# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.layers.merge import Concatenate
from keras.layers import concatenate
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SEQUENCE_LENGTH_Q = 1000
MAX_SEQUENCE_LENGTH_A = 1000
MAX_SEQUENCE_LENGTH_C = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv('datum.txt', sep='\t')
print(data_train.shape)

texts_q = []
texts_a = []
texts_c = []
labels = []

for idx in range(data_train.question.shape[0]):
    text = BeautifulSoup(data_train.question[idx])
    texts_q.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    text = BeautifulSoup(data_train.answer[idx])
    texts_a.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    text = BeautifulSoup(data_train.context[idx])
    texts_c.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    labels.append(data_train.value[idx])

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_q + texts_c + texts_a)
sequences_q = tokenizer.texts_to_sequences(texts_q)
sequences_a = tokenizer.texts_to_sequences(texts_a)
sequences_c = tokenizer.texts_to_sequences(texts_c)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_q = pad_sequences(sequences_q, maxlen=MAX_SEQUENCE_LENGTH_Q)
data_a = pad_sequences(sequences_a, maxlen=MAX_SEQUENCE_LENGTH_A)
data_c = pad_sequences(sequences_c, maxlen=MAX_SEQUENCE_LENGTH_C)

labels = to_categorical(np.asarray(labels))
print('Shape of data q tensor:', data_q.shape)
print('Shape of data a tensor:', data_a.shape)
print('Shape of data c tensor:', data_c.shape)
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

GLOVE_DIR = "~/Testground/data/glove"
embeddings_index = {}
f = open('glove.6B.300d.txt', 'r', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH_C,
                            trainable=False)

context = Input(shape=(MAX_SEQUENCE_LENGTH_Q,), dtype='int32', name='context')
question = Input(shape=(MAX_SEQUENCE_LENGTH_A,), dtype='int32', name='question')
answer = Input(shape=(MAX_SEQUENCE_LENGTH_C,), dtype='int32', name='answer')
embedded_context = embedding_layer(context)
embedded_question = embedding_layer(question)
embedded_answer = embedding_layer(answer)
l_lstm_c = Bidirectional(LSTM(100))(embedded_context)
l_lstm_q = Bidirectional(LSTM(30))(embedded_question)
l_lstm_a = Bidirectional(LSTM(10))(embedded_answer)

concat_c_q = concatenate([l_lstm_c, l_lstm_q], axis = 1)
softmax_c_q = Dense(EMBEDDING_DIM, activation='softmax')(concat_c_q)

concat_c_q_a = concatenate([l_lstm_a, softmax_c_q], axis = 1)

softmax_c_q = Dense(2, activation='softmax')(concat_c_q_a)
model = Model([context, question, answer], softmax_c_q)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
model.summary()
model.fit({'context': context_data, 'question': question_data, 'answer': answer_data}, y_train,
          validation_data=({'context': context_data_v, 'question': question_data_v, 'answer': answer_data_v}, y_val),
          nb_epoch=5, batch_size=20)

