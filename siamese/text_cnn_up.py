import tensorflow as tf
import numpy as np
import gensim, logging
from gensim import corpora, models, similarities

class TextCNNUp(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, dictionary, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        def embedChars(input):
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.embedded_chars = tf.nn.embedding_lookup(dictionary, input)
                return tf.expand_dims(tf.to_float(self.embedded_chars), -1)

        self.embedded_chars_1 = embedChars(self.input_x1)
        self.embedded_chars_2 = embedChars(self.input_x2)

        # Create a convolution + maxpool layer for each filter size
        def convolution_1(input):
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, 1, 1, num_filters]
                    W = tf.get_variable("W", filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(
                        input,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, filter_sizes[len(filter_sizes) - i - 1], 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            return pooled_outputs
        # Create a convolution + maxpool layer for each filter size
        def convolution_2(input, num_channels, num_filters):
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-2-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, 300, num_channels, num_filters]
                    W = tf.get_variable("W", filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(
                        input,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Avgpooling over the outputs
                    aux = int((filter_sizes[len(filter_sizes) - 1] + filter_sizes[0]) / 2) - 1
                    pooled = tf.nn.avg_pool(
                        h,
                        ksize=[1, sequence_length - 2 * aux - (filter_size - 1), 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            return pooled_outputs

        data = []
        with tf.variable_scope("convolutions") as scope:
            conv1 = convolution_1(self.embedded_chars_1)
            data.append(conv1)
        with tf.variable_scope("convolutions", reuse=True) as scope:
            conv2 = convolution_1(self.embedded_chars_2)
            data.append(conv2)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool0 = tf.concat(3, data[0])
        #self.h_pool_flat0 = tf.reshape(self.h_pool0, [-1, num_filters_total])
        self.h_pool1 = tf.concat(3, data[1])
        #self.h_pool_flat1 = tf.reshape(self.h_pool1, [-1, num_filters_total])

        with tf.variable_scope("convolutions_2") as scope:
            conv1 = convolution_2(self.h_pool0, num_filters_total, num_filters_total * 2)
        with tf.variable_scope("convolutions_2", reuse=True) as scope:
            conv2 = convolution_2(self.h_pool1, num_filters_total, num_filters_total * 2)

        pooled_outputs = conv1 + conv2
        # Combine all the pooled features
        num_filters_total = num_filters_total * 2 * 6
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
