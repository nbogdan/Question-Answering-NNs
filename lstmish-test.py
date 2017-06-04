# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input
from keras.layers import Embedding, Dropout, LSTM, Bidirectional
from keras.layers import concatenate
from keras.models import Model

import pickle
import keras, h5py
from siamese.data_helpers import get_lemmas
import re
from nltk.stem import WordNetLemmatizer


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


MAX_SEQUENCE_LENGTH_Q = 400
MAX_SEQUENCE_LENGTH_A = 400
MAX_SEQUENCE_LENGTH_C = 400
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

data_test = pd.read_csv('test_datum_.txt', sep='\t')
print(data_test.shape)

labels = []
texts_q = []
texts_a = []
texts_c = []
for idx in range(data_test.question.shape[0]):
    text = BeautifulSoup(data_test.question[idx])
    texts_q.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    text = BeautifulSoup(data_test.answer[idx])
    texts_a.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    text = BeautifulSoup(data_test.context[idx])
    texts_c.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    labels.append(data_test.value[idx])

lemmatizer = WordNetLemmatizer()
texts_c3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_c]
texts_q3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_q]
texts_a3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_a]

with open('test_outfile_c_', 'wb') as fp:
    pickle.dump(texts_c3, fp)
with open('test_outfile_q_', 'wb') as fp:
    pickle.dump(texts_q3, fp)
with open('test_outfile_a_', 'wb') as fp:
    pickle.dump(texts_a3, fp)
texts_c3 = pickle.load(open('test_outfile_c_', 'rb'))
texts_q3 = pickle.load(open('test_outfile_q_', 'rb'))
texts_a3 = pickle.load(open('test_outfile_a_', 'rb'))

# tokenizer = pickle.load(open('tokenizer', 'rb'))
# sequences_q = tokenizer.texts_to_sequences(texts_q3)
# sequences_a = tokenizer.texts_to_sequences(texts_a3)
# sequences_c = tokenizer.texts_to_sequences(texts_c3)
#
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
#
# data_q = pad_sequences(sequences_q, maxlen=MAX_SEQUENCE_LENGTH_Q)
# data_a = pad_sequences(sequences_a, maxlen=MAX_SEQUENCE_LENGTH_A)
# data_c = pad_sequences(sequences_c, maxlen=MAX_SEQUENCE_LENGTH_C)
#
# labels = to_categorical(np.asarray(labels))
# print('Shape of data q tensor:', data_q.shape)
# print('Shape of data a tensor:', data_a.shape)
# print('Shape of data c tensor:', data_c.shape)
# print('Shape of label tensor:', labels.shape)
#
# indices = np.arange(data_q.shape[0])
# # np.random.shuffle(indices)
# # data_q = data_q[indices]
# # data_a = data_a[indices]
# # data_c = data_c[indices]
# # labels = labels[indices]
# question_data = data_q
# answer_data = data_a
# context_data = data_c
# y_test = labels
#
# print('Traing and validation set number of positive and negative reviews')
# print(y_test.sum(axis=0))
#
# embedding_matrix = pickle.load(open('embedding_matrix', 'rb'))
#
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH_C,
#                             trainable=False)
#
# context = Input(shape=(MAX_SEQUENCE_LENGTH_Q,), dtype='int32', name='context')
# question = Input(shape=(MAX_SEQUENCE_LENGTH_A,), dtype='int32', name='question')
# answer = Input(shape=(MAX_SEQUENCE_LENGTH_C,), dtype='int32', name='answer')
# embedded_context = embedding_layer(context)
# embedded_question = embedding_layer(question)
# embedded_answer = embedding_layer(answer)
# l_lstm_c = Bidirectional(LSTM(50))(embedded_context)
# l_lstm_q = Bidirectional(LSTM(30))(embedded_question)
# l_lstm_a = Bidirectional(LSTM(20))(embedded_answer)
#
# concat_c_q = concatenate([l_lstm_c, l_lstm_q], axis = 1)
# softmax_c_q = Dense(EMBEDDING_DIM, activation='softmax')(concat_c_q)
#
# concat_c_q_a = concatenate([l_lstm_a, softmax_c_q], axis = 1)
# drop = Dropout(0.1)(concat_c_q_a)
#
# softmax_c_q = Dense(2, activation='softmax')(drop)
# model = Model([context, question, answer], softmax_c_q)
# model.load_weights('simpleData-27-0.89.hdf5') #27 - 089 - 41%
# opt = keras.optimizers.Nadam()
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['acc'])
#
# result = model.evaluate({'context': context_data, 'question': question_data, 'answer': answer_data}, y_test,
#           batch_size=512)
# print(result)
#
# result2 = model.predict({'context': context_data, 'question': question_data, 'answer': answer_data}, batch_size=300)
# result2_c = [np.argmax(x) for x in result2]
# real_c = [np.argmax(x) for x in y_test]
#
# correct = 0
# for i in range(int(len(real_c) / 4)):
#     results = result2[i*4:i*4+4]
#     real_result = np.argmax(real_c[i*4:i*4+4])
#     aux = np.argmax([x[1] for x in results])
#     if aux == real_result:
#         correct+=1
# print(correct)

# true_found = 0
# false_found = 0
# for i in range(len(result2_c)):
#     if result2_c[i] == real_c[i] and result2_c[i] == 1:
#         true_found += 1
#     if result2_c[i] == real_c[i] and result2_c[i] == 0:
#         false_found += 1
#
# print(result2)
# print('True found: ' + str(true_found))
# print('False found: ' + str(false_found))
# print('True recall: ' + str(true_found / 300))
# print('True precision: ' + str(true_found / (true_found + 300 - false_found)))
# print('False recall: ' + str(false_found / 300))
# print('False precision: ' + str(false_found / (false_found + 300 - true_found)))