from keras.layers import Embedding, Dropout, Dense, Input, Convolution1D, MaxPooling1D, Flatten, Concatenate, merge
from keras.models import Model
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import ModelCheckpoint

MAX_SEQUENCE_LENGTH_Q = 100
MAX_SEQUENCE_LENGTH_A = 20
MAX_SEQUENCE_LENGTH_C = 500
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
VERSION = "1"

class CosCNN():
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

        conv_blocksA = []
        conv_blocksQ = []
        for sz in [3,5]:
            conv = Convolution1D(filters=20,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(embedded_answer)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocksA.append(conv)
        for sz in [5,7]:
            conv = Convolution1D(filters=20,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(embedded_question)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocksQ.append(conv)

        conv_c = Convolution1D(filters=40,
                                 kernel_size=9,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(embedded_context)
        conv_c = MaxPooling1D(pool_size=2)(conv_c)
        conv_c = Flatten()(conv_c)

        z_q = Concatenate()(conv_blocksA + [conv_c])
        z_a = Concatenate()(conv_blocksQ + [conv_c])
        z_q = Dropout(0.5)(z_q)
        z_a = Dropout(0.5)(z_a)
        z_q = Dense(100, activation="relu")(z_q)
        z_a = Dense(100, activation="relu")(z_a)
        concat_c_q_a = merge([z_q, z_a], mode='cos')
        softmax_c_q = Dense(2, activation='softmax')(concat_c_q_a)
        self.model = Model([context, question, answer], softmax_c_q)
        opt = Nadam()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['acc'])

    def train(self, train_data, validation_data, folder):
        context_data, question_data, answer_data, y_train = train_data
        context_data_v, question_data_v, answer_data_v, y_val = validation_data
        print("Model Fitting")
        filepath = folder + "structures/cos-cnn" + VERSION + "-final-{epoch:02d}-{val_acc:.2f}.hdf5"

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        model_json = self.model.to_json()
        with open(folder + "/structures/cos-cnn-model" + VERSION + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.summary()
        self.model.fit({'context': context_data, 'question': question_data, 'answer': answer_data}, y_train,
                  validation_data=({'context': context_data_v, 'question': question_data_v, 'answer': answer_data_v}, y_val),
                  epochs=50, batch_size=256, callbacks=[checkpoint])
