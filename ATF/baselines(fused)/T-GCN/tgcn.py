# -*- coding: utf-8 -*-

from inits import *
from utils import calculate_laplacian

class tgcnCell(tf.nn.rnn_cell.GRUCell):
    """Temporal Graph Convolutional Network """

    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):

        super(tgcnCell, self).__init__(num_units=num_units)
        self._act = act
        self._nodes = num_nodes  # 站点个数
        self._units = num_units  # 隐藏状态大小
        self._adj = []
        self._adj.append(calculate_laplacian(adj))


    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "tgcn"):
            with tf.variable_scope("gates"):
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
        return new_h, new_h

    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        # inputs: (batch, num_nodes, feature_dim)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        x_s = tf.concat([inputs, state], axis=2)  # (batch, num_nodes, feature_dim + hidden)
        input_size = x_s.get_shape()[2].value

        # 图卷积
        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            for m in self._adj:
                x1 = tf.sparse_tensor_dense_matmul(m, x0)
            x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
            x = tf.transpose(x, perm=[2, 0, 1])
            x = tf.reshape(x, shape=[-1, input_size])

            weights = tf.get_variable('weights', [input_size, output_size],
                                      initializer=tf.initializers.truncated_normal())
            biases = tf.get_variable("biases", [output_size],
                                     initializer=tf.constant_initializer(bias, dtype=tf.float32))
            x = tf.matmul(x, weights)
            x = tf.nn.bias_add(x, biases)

            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x

