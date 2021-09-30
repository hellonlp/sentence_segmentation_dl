# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:01:45 2019

@author: cm
"""

import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper


def cell_bilstm(inputs, hidden_size, is_training, name_scope="cell_bilstm"):
    """
    inputs shape: (batch_size,sequence_length,embedding_size)
    hidden_size: rnn hidden size
    """
    with tf.variable_scope(name_scope):
        cell_forward = tf.contrib.rnn.BasicLSTMCell(hidden_size / 2)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(hidden_size / 2)
        cell_forward = DropoutWrapper(cell_forward,
                                      input_keep_prob=1.0,
                                      output_keep_prob=0.5 if is_training else 1)
        cell_backward = DropoutWrapper(cell_backward,
                                       input_keep_prob=1.0,
                                       output_keep_prob=0.5 if is_training else 1)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_forward,
                                                          cell_backward,
                                                          inputs,
                                                          dtype=tf.float32)
        forward_out, backward_out = outputs
        # Concat
        outputs = tf.concat([forward_out, backward_out], axis=2)
        # Activation
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)
    return outputs


def cell_dense(inputs, num_labels, name_scope="Full-connection"):
    with tf.name_scope(name_scope):
        outputs = tf.layers.dense(inputs,
                                  units=num_labels,
                                  activation=tf.nn.relu,
                                  name="dense1")
    return outputs
