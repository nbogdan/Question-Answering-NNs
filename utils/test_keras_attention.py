'''
    Module used for playing with the Keras custom Attention Layer.
    It uses the IMDb dataset in order to perform sentiment analysis
    over a set of reviews (about 25000 reviews).
'''
import numpy as np

from keras.datasets import imdb
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras_utils import AttentionLayerV2


# Prints the top 10 words as selected by the attention.
# @param model (Keras.model): a pretrained model.
# @param in_seq (list(int)): a sequence as defined in the IMDb dataset.
def highlight_attention_words(model, in_seq):
    aux = imdb.get_word_index()
    words = {}
    for x in aux:
        words[aux[x]] = x
    sentence = ''
    for x in in_seq:
        if x >= 3 and x - 3 in words:
            sentence += words[x - 3] + ' '
    print(sentence)
    print("")
    print("Top worlds:")
    x = model.predict(np.expand_dims(in_seq, 0))[0]
    sol = []
    for i in range(0, len(in_seq)):
        if in_seq[i] >= 3:
            sol.append((x[i], in_seq[i]))
    sol.sort(reverse=True)
    for i in range(0, 10):
        score = sol[i][0]
        x = sol[i][1]
        print(words[x - 3], score)


def main():
    top_words = 5000  # Keep only the most frequent 500 words in the dataset.
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # Keras requires same length (although 0 will mean no information).
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    embedding_length = 32
    input_seq = Input(shape=(500,))
    a = Embedding(top_words, embedding_length,
                  input_length=max_review_length)(input_seq)
    b, state_h, state_c = LSTM(100, return_state=True,
                               return_sequences=True)(a)
    c = AttentionLayerV2(attention_depth=4)(b)
    d = Dropout(0.5)(c)
    e = Dense(1, activation='sigmoid')(d)
    model = Model(inputs=[input_seq], outputs=[e])
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    model.summary()
    # print(model.predict(np.ones((10, 500))))
    model.fit(X_train, y_train, epochs=5, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    model.save_weights('model_weights.h5')
    # model.load_weights('model_weights.h5', by_name=True)
    # highlight_attention_words(model, X_test[11])


if __name__ == "__main__":
    main()
