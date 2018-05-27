import tensorflow as tf
from .nn_visualization import variable_summaries


def simple_layer(name, input_data, shape, activation='linear'):
    w_name = 'w_' + name
    b_name = 'b_' + name
    if activation == 'relu':
        w = tf.get_variable(w_name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    else:
        w = tf.get_variable(w_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(b_name, initializer=tf.constant_initializer(0.), shape=shape[1])

    variable_summaries(w_name + 'summary', w)
    variable_summaries(b_name + 'summary', bias)

    output_data = tf.matmul(input_data, w) + bias
    if activation == 'relu':
        output_data = tf.nn.relu(output_data)
    elif activation == 'sigmoid':
        output_data = tf.nn.sigmoid(output_data)
    elif activation == 'tanh':
        output_data = tf.nn.tanh(output_data)
    return output_data
