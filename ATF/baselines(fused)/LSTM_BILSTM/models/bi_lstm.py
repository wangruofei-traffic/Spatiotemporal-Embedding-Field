# -- coding: utf-8 --
from models.inits import *

class BilstmClass(object):
    def __init__(self, hp, placeholders=None):
        '''
        :param hp:
        :param placeholders:
        '''
        self.hp = hp
        self.batch_size = self.hp.batch_size
        self.layer_num = self.hp.hidden_layer
        self.hidden_size = self.hp.hidden_size
        self.input_length = self.hp.input_length
        self.output_length = self.hp.output_length
        self.placeholders = placeholders
        self.encoder()
        self.decoder()

    def lstm(self):
        def cell():
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)
            lstm_cell_ = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,output_keep_prob=1-self.placeholders['dropout'])
            return lstm_cell_
        mlstm = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.layer_num)])
        return mlstm

    def bilstm(self):
        def cell():
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)  # single lstm unit
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=1-self.placeholders['dropout'])
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_size)  # single lstm unit
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=1-self.placeholders['dropout'])
            return cell_fw, cell_bw
        cell_fw, cell_bw=cell()
        f_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell_fw for _ in range(self.layer_num)])
        b_mlstm = tf.nn.rnn_cell.MultiRNNCell([cell_bw for _ in range(self.layer_num)])
        return f_mlstm, b_mlstm

    def encoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''
        self.e_lstm_1 = self.lstm()
        self.ef_bilstm_2, self.eb_bilstm_2 = self.bilstm()
        self.e_lstm_3 = self.lstm()

    def decoder(self):
        '''
        :return:
        '''
        self.d_lstm_1 = self.lstm()
        self.df_bilstm_2, self.db_bilstm_2 = self.bilstm()
        self.d_lstm_3 = self.lstm()

    def encoding(self, inputs):
        #print("inputs: ", inputs.shape)
        '''
        :param inputs:
        :return: shape is [batch size, time size, hidden size]
        '''
        with tf.variable_scope('feature_mapping'):
            # 第一个特征（维度1）
            first_feature = inputs[:, :, 0:1]  # [B, T, 1]
            # 经过全连接层映射为32维
            first_mapped = tf.layers.dense(first_feature, units=32, activation=None, name='first_map')

            # 取剩下的32维
            rest_features = inputs[:, :, 1:]  # [B, T, 32]
            # 拼接得到64维
            x = tf.concat([first_mapped, rest_features], axis=-1)  # [B, T, 64]
            #print("x",x.shape)

        with tf.variable_scope('encoder_lstm_1'):
            lstm_1_outpus, _ = tf.nn.dynamic_rnn(cell=self.e_lstm_1, inputs=x, dtype=tf.float32)
            x = lstm_1_outpus
        with tf.variable_scope('encoder_bilstm_2'):
            bilstm_2_outpus, _ = tf.nn.bidirectional_dynamic_rnn(self.ef_bilstm_2, self.eb_bilstm_2, x, dtype=tf.float32)
            # shape is [2, batch_size, seq_length, output_size]
            x = tf.concat(bilstm_2_outpus, axis=2)
            x = tf.layers.dense(inputs=x, units=self.hidden_size, activation=None,name='encoder_full')
        with tf.variable_scope('encoder_lstm_3'):
            lstm_3_outpus,_ = tf.nn.dynamic_rnn(cell=self.e_lstm_3, inputs=x, dtype=tf.float32)
            x = lstm_3_outpus
        return x

    def decoding(self, encoder_hs, site_num, future_attr):
        """
        :param encoder_hs: 编码器输出 [B*site, input_len, hidden_size]
        :param site_num: 站点数
        :param future_attr: 标签未来属性部分 [B*site, output_len, 32]
        """

        #print("encoder_hs:", encoder_hs.shape)
        #print("future_attr:", future_attr.shape)

        with tf.variable_scope('future_attr_mapping', reuse=tf.AUTO_REUSE):
            future_attr_emb = tf.layers.dense(future_attr, units=self.hidden_size, activation=tf.nn.relu,
                                              name='attr_fc')
            # [B*site, output_len, hidden_size]
            #print("future_attr_emb",future_attr_emb.shape)

        pres = []
        h_state = encoder_hs[:, -1:, :]  # 取编码器最后一个时刻的隐藏状态

        for i in range(self.output_length):
            with tf.variable_scope(f'decoder_step_{i}', reuse=tf.AUTO_REUSE):
                attr_t = future_attr_emb[:, i:i + 1, :]  # 当前步属性
                dec_input = tf.concat([h_state, attr_t], axis=-1)  # 拼接当前输入
                #print("dec_input", dec_input.shape)

                # === 三层解码结构 ===
                with tf.variable_scope('decoder_lstm_1', reuse=tf.AUTO_REUSE):
                    lstm_1_out, _ = tf.nn.dynamic_rnn(self.d_lstm_1, dec_input, dtype=tf.float32)
                    x = lstm_1_out

                with tf.variable_scope('decoder_bilstm_2', reuse=tf.AUTO_REUSE):
                    bilstm_out, _ = tf.nn.bidirectional_dynamic_rnn(self.df_bilstm_2, self.db_bilstm_2, x,
                                                                    dtype=tf.float32)
                    x = tf.concat(bilstm_out, axis=2)

                with tf.variable_scope('decoder_lstm_3', reuse=tf.AUTO_REUSE):
                    lstm_3_out, _ = tf.nn.dynamic_rnn(self.d_lstm_3, x, dtype=tf.float32)
                    h_state = lstm_3_out[:, -1:, :]  # 更新隐藏状态

                #print("h_state", h_state.shape)

                # === 输出层 ===
                layer_1 = tf.layers.dense(tf.squeeze(h_state, axis=1), units=64, activation=tf.nn.relu, name='layer1',
                                          reuse=tf.AUTO_REUSE)
                results = tf.layers.dense(layer_1, units=1, name='layer2', reuse=tf.AUTO_REUSE)

                pre = tf.reshape(results, [-1, site_num])
                pres.append(tf.expand_dims(pre, axis=-1))

        return tf.concat(pres, axis=-1, name='output_y')


import numpy as np
if __name__ == '__main__':
    train_data=np.random.random(size=[32,3,16])
    x=tf.placeholder(tf.float32, shape=[32, 3, 16])
    r=lstm(32,10,2,128)
    hs=r.encoding(x)

    print(hs.shape)

    pre=r.decoding(hs)
    print(pre.shape)