# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, Input
from keras.layers import Embedding, Dropout, LSTM, Bidirectional
from keras.layers import concatenate
from keras.models import Model

import keras
from siamese.data_helpers import get_lemmas
import re, string
from nltk.stem import WordNetLemmatizer
import pickle

MAX_SEQUENCE_LENGTH_Q = 400
MAX_SEQUENCE_LENGTH_A = 400
MAX_SEQUENCE_LENGTH_C = 400
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv('train_datum_.txt', sep='\t')
data_train2 = pd.read_csv('train_datum2.txt', sep='\t', error_bad_lines=False)
print(data_train.shape)
print(data_train2.shape)

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

for idx in range(data_train2.question.shape[0]):
    text = BeautifulSoup(data_train2.question[idx])
    texts_q.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    text = BeautifulSoup(data_train2.answer[idx])
    texts_a.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    text = BeautifulSoup(str(data_train2.context[idx]))
    texts_c.append(clean_str(str(text.get_text().encode('ascii', 'ignore')))[1:])
    labels.append(data_train2.value[idx])

print('Soup is done')
lemmatizer = WordNetLemmatizer()
texts_c3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_c]
texts_q3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_q]
texts_a3 = [" ".join(get_lemmas(re.findall(r'\b\w+\b', s), lemmatizer)) for s in texts_a]

with open('outfile_c2', 'wb') as fp:
    pickle.dump(texts_c3, fp)
with open('outfile_q2', 'wb') as fp:
    pickle.dump(texts_q3, fp)
with open('outfile_a2', 'wb') as fp:
    pickle.dump(texts_a3, fp)
print('Saved lemmas1')

texts_c3 = pickle.load(open('outfile_c2', 'rb'))
texts_q3 = pickle.load(open('outfile_q2', 'rb'))
texts_a3 = pickle.load(open('outfile_a2', 'rb'))
test_texts_c3 = pickle.load(open('test_outfile_c_', 'rb'))
test_texts_q3 = pickle.load(open('test_outfile_q_', 'rb'))
test_texts_a3 = pickle.load(open('test_outfile_a_', 'rb'))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_q3 + texts_c3 + texts_a3 + test_texts_a3 + test_texts_c3 + test_texts_q3)
print('Saving tokenizer')
with open('tokenizer2', 'wb') as fp:
    pickle.dump(tokenizer, fp)
tokenizer = pickle.load(open('tokenizer2', 'rb'))
sequences_q = tokenizer.texts_to_sequences(texts_q3)
sequences_a = tokenizer.texts_to_sequences(texts_a3)
sequences_c = tokenizer.texts_to_sequences(texts_c3)

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

# print('Saving embedding matrix')
with open('embedding_matrix2', 'wb') as fp:
    pickle.dump(embedding_matrix, fp)
embedding_matrix = pickle.load(open('embedding_matrix2', 'rb'))

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
l_lstm_c = Bidirectional(LSTM(50))(embedded_context)
l_lstm_q = Bidirectional(LSTM(30))(embedded_question)
l_lstm_a = Bidirectional(LSTM(20))(embedded_answer)

concat_c_q = concatenate([l_lstm_c, l_lstm_q], axis = 1)
softmax_c_q = Dense(EMBEDDING_DIM, activation='softmax')(concat_c_q)

concat_c_q_a = concatenate([l_lstm_a, softmax_c_q], axis = 1)
drop = Dropout(0.1)(concat_c_q_a)

softmax_c_q = Dense(2, activation='softmax')(drop)
model = Model([context, question, answer], softmax_c_q)
opt = keras.optimizers.Nadam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

print("model fitting - Bidirectional LSTM")
filepath="allData-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.summary()
model.fit({'context': context_data, 'question': question_data, 'answer': answer_data}, y_train,
          validation_data=({'context': context_data_v, 'question': question_data_v, 'answer': answer_data_v}, y_val),
          nb_epoch=40, batch_size=512, callbacks=[checkpoint])

