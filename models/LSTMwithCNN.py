from keras.layers import Embedding, Dropout, Dense, Input, Convolution1D, MaxPooling1D, Flatten, Concatenate
from keras.models import Model
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dropout, LSTM, Bidirectional
from keras.layers import concatenate
import keras

MAX_SEQUENCE_LENGTH_Q = 100
MAX_SEQUENCE_LENGTH_A = 20
MAX_SEQUENCE_LENGTH_C = 500
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
VERSION = "1"

class LSTMwithCNN():
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

        l_lstm_c = Bidirectional(LSTM(60, return_sequences=True))(embedded_context)
        conv_blocksC = []
        for sz in [5,7]:
            conv = Convolution1D(filters=20,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(l_lstm_c)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocksC.append(conv)

        l_lstm_q = Bidirectional(LSTM(60, return_sequences=True))(embedded_question)
        conv_blocksQ = []
        for sz in [3, 5]:
            conv = Convolution1D(filters=20,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(l_lstm_q)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocksQ.append(conv)
        l_lstm_a = Bidirectional(LSTM(60))(embedded_answer)

        concat_c_q = concatenate([l_lstm_a] + conv_blocksQ + conv_blocksC , axis=1)
        relu_c_q_a = Dense(100, activation='relu')(concat_c_q)
        relu_c_q_a = Dropout(0.25)(relu_c_q_a)
        softmax_c_q_a = Dense(2, activation='softmax')(relu_c_q_a)
        self.model = Model([question, answer, context], softmax_c_q_a)
        opt = Nadam()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['acc'])

    def train(self, train_data, validation_data, folder):
        context_data, question_data, answer_data, y_train = train_data
        context_data_v, question_data_v, answer_data_v, y_val = validation_data
        print("Model Fitting")
        filepath = folder + "structures/lstm-nn" + VERSION + "-final-{epoch:02d}-{val_acc:.2f}.hdf5"

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        model_json = self.model.to_json()
        with open(folder + "/structures/lstm-model" + VERSION + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.summary()
        self.model.fit({'context': context_data, 'question': question_data, 'answer': answer_data}, y_train,
                  validation_data=({'context': context_data_v, 'question': question_data_v, 'answer': answer_data_v}, y_val),
                  epochs=100, batch_size=256, callbacks=[checkpoint])
