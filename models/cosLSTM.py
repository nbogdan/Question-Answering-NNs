from keras.layers import Embedding, Dropout, Dense, Input, Convolution1D, MaxPooling1D, Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dropout, LSTM, Bidirectional
from keras.layers import concatenate, merge
import keras

MAX_SEQUENCE_LENGTH_Q = 100
MAX_SEQUENCE_LENGTH_A = 20
MAX_SEQUENCE_LENGTH_C = 500
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
VERSION = "1"

class CosLSTM():
    def __init__(self, word_index, embedding_matrix):
        embedding_layer_c = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH_C,
                                    trainable=False)
        embedding_layer_q = Embedding(len(word_index) + 1,
                                      EMBEDDING_DIM,
                                      weights=[embedding_matrix],
                                      input_length=MAX_SEQUENCE_LENGTH_Q,
                                      trainable=False)
        embedding_layer_a = Embedding(len(word_index) + 1,
                                      EMBEDDING_DIM,
                                      weights=[embedding_matrix],
                                      input_length=MAX_SEQUENCE_LENGTH_A,
                                      trainable=False)
        context = Input(shape=(MAX_SEQUENCE_LENGTH_C,), dtype='int32', name='context')
        question = Input(shape=(MAX_SEQUENCE_LENGTH_Q,), dtype='int32', name='question')
        answer = Input(shape=(MAX_SEQUENCE_LENGTH_A,), dtype='int32', name='answer')
        embedded_context = embedding_layer_c(context)
        embedded_question = embedding_layer_q(question)
        embedded_answer = embedding_layer_a(answer)

        l_lstm_c = Bidirectional(LSTM(60))(embedded_context)
        l_lstm_q = Bidirectional(LSTM(60))(embedded_question)
        l_lstm_a = Bidirectional(LSTM(60))(embedded_answer)

        concat_c_q = concatenate([l_lstm_q, l_lstm_c], axis=1)
        relu_c_q = Dense(100, activation='tanh')(concat_c_q)
        concat_c_a = concatenate([l_lstm_a, l_lstm_c], axis=1)
        relu_c_a = Dense(100, activation='tanh')(concat_c_a)
        relu_c_q = Dropout(0.5)(relu_c_q)
        relu_c_a = Dropout(0.5)(relu_c_a)
        concat_c_q_a = merge([relu_c_a, relu_c_q], mode='cos')
        softmax_c_q_a = Dense(2, activation='softmax')(concat_c_q_a)
        self.model = Model([question, answer, context], softmax_c_q_a)
        opt = Nadam()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['acc'])

    def train(self, train_data, validation_data, folder):
        context_data, question_data, answer_data, y_train = train_data
        context_data_v, question_data_v, answer_data_v, y_val = validation_data
        print("Model Fitting")
        filepath = folder + "structures/cos-lstm-nn" + VERSION + "-final-{epoch:02d}-{val_acc:.2f}.hdf5"

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        model_json = self.model.to_json()
        with open(folder + "/structures/cos-lstm-model" + VERSION + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.summary()
        import numpy as np
        context_data = np.array(list(map(lambda x: x[:MAX_SEQUENCE_LENGTH_C], context_data)))
        context_data_v = np.array(list(map(lambda x: x[:MAX_SEQUENCE_LENGTH_C], context_data_v)))
        self.model.fit({'context': context_data, 'question': question_data, 'answer': answer_data}, y_train,
                  validation_data=({'context': context_data_v, 'question': question_data_v, 'answer': answer_data_v}, y_val),
                  epochs=50, batch_size=256, callbacks=[checkpoint], verbose=2)
