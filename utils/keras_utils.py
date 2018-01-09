'''
    Some extra util functions and custom layers defined over
    the Keras API. Some of them are only compatible with specific
    backends, like TensorFlow or Theano.
'''
from keras import initializers
from keras import backend as K
from keras.engine import Layer


# Defines an attention layer over a RNN as specified in [1].
# To be used immediately over any RNN such as LSTM or GRU cell.
# It needs access to internal outputs at every timestep. Therefore,
# one must set return_state and return_sequences when building the
# RNN cell (see Keras doc [2]).
# Example of use:
#       b, _, _ = LSTM(100, return_state=True, return_sequences=True)(prev)
#       c = AttentionLayer()(b) -> output [batch_size, 100]
# It can be used only with TensorFlow as Keras backend.
# The output from the layer has the shape of one hidden state in the RNN.
# [1] https://arxiv.org/pdf/1512.08756.pdf
# [2] https://keras.io/layers/recurrent/
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert(len(input_shape) == 3)  # (batches, time_steps, hidden_nodes)
        self.W = self.add_weight(name='AttentionLayer_weight1',
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 trainable=True)
        self.bias = self.add_weight(name='AttentionLayer_bias1',
                                    shape=(input_shape[1],),
                                    initializer='zero',
                                    trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def call(self, x):
        assert(K.backend() == 'tensorflow')
        # The model is described by the following equations:
        #       estimated_weight_i = dot_product(hidden_state_i, W).
        #       biased_weight = estimated_weight + bias
        #       non_linear_weight = tanh(biased_weight)
        estimated_weight = K.squeeze(K.dot(x, K.expand_dims(self.W, -1)), -1)
        biased_weight = estimated_weight + self.bias
        non_linear_weight = K.tanh(biased_weight)

        # For each hidded state calculate how much should it contribute
        # to the context vector. This is the main part of attention.
        # In order to convert weights to "probabilities" use a sigmoid
        # based function: exp(x) / sum(exp(xi)).
        prob = K.exp(non_linear_weight)
        # Compute the total sum for each batch.
        total_sum = K.sum(prob, axis=1, keepdims=True)
        prob /= K.cast(total_sum, K.floatx())

        # Multiply each hidden value by the corresponding probability.
        prob = K.expand_dims(prob, -1)
        new_hidden_values = x * prob
        return K.sum(new_hidden_values, axis=1)


# Simiar to AttentionLayer. The difference is that the function
# that assigns weights to each hidden state in the RNN is learned
# via a deeper neural network (a number of k Dense layers with
# biases at every layer). It should be more accurate but it takes
# more time to train (than AttentionLayer) and much more memory.
class AttentionLayerV2(Layer):
    def __init__(self, attention_depth=2, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.attention_depth = attention_depth
        super(AttentionLayerV2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert(len(input_shape) == 3)  # (batches, time_steps, hidden_nodes)
        self.Ws = []  # Weights at each layer.
        self.bs = []  # Biases at each layer.
        for i in range(0, self.attention_depth):
            W = self.add_weight(name='AttentionLayer_weight{}'.format(i),
                                shape=(input_shape[1], input_shape[1]),
                                initializer=self.init,
                                trainable=True)
            b = self.add_weight(name='AttentionLayer_bias{}'.format(i),
                                shape=(input_shape[1],),
                                initializer='zero',
                                trainable=True)
            self.Ws.append(W)
            self.bs.append(b)
        # Final layer weights and biases.
        self.Wf = self.add_weight(name='AttentionLayer_weightf',
                                  shape=(input_shape[-1],),
                                  initializer=self.init,
                                  trainable=True)
        self.bias = self.add_weight(name='AttentionLayer_biasf',
                                    shape=(input_shape[1],),
                                    initializer='zero',
                                    trainable=True)
        super(AttentionLayerV2, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    # A forward step in the layer.
    def call(self, x):
        assert(K.backend() == 'tensorflow')
        temp = K.permute_dimensions(x, (0, 2, 1))
        for i in range(0, self.attention_depth):
            temp = K.sigmoid(K.dot(temp, self.Ws[i]) + self.bs[i])
        temp = K.permute_dimensions(temp, (0, 2, 1))
        estimated_weight = K.squeeze(K.dot(temp, K.expand_dims(self.Wf, -1)), -1)
        biased_weight = estimated_weight + self.bias
        non_linear_weight = K.tanh(biased_weight)

        # For each hidded state calculate how much should it contribute
        # to the context vector. This is the main part of attention.
        # In order to convert weights to "probabilities" use a sigmoid
        # based function: exp(x) / sum(exp(xi)).
        prob = K.exp(non_linear_weight)
        # Compute the total sum for each batch.
        total_sum = K.sum(prob, axis=1, keepdims=True)
        prob /= K.cast(total_sum, K.floatx())

        # Enable this if you want access to internal probabilities.
        # Should only be used for testing that Attention works as expected.
        # return prob

        # Multiply each hidden value by the corresponding probability.
        prob = K.expand_dims(prob, -1)
        new_hidden_values = x * prob
        return K.sum(new_hidden_values, axis=1)
