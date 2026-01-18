# -- coding: utf-8 --

from __future__ import division
from __future__ import print_function

from tgcn import tgcnCell
from utils import *
from hyparameter import parameter
import matplotlib.pyplot as plt
from data_load import *
from inits import *

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

os.environ['CUDA_VISIBLE_DEVICES']='0'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


class Model(object):
    def __init__(self, para, mean, std):
        '''
        :param para:
        '''
        self.para = para
        self.mean = mean
        self.std = std
        self.input_len = self.para.input_length
        self.output_len = self.para.output_length
        self.total_len = self.input_len + self.output_len
        self.features = self.para.features
        self.batch_size = self.para.batch_size
        self.epochs = self.para.epoch
        self.site_num = self.para.site_num
        self.emb_size = self.para.emb_size
        self.hidden_size = self.para.hidden_size
        self.is_training = self.para.is_training
        self.learning_rate = self.para.learning_rate
        self.model_name = self.para.model_name
        self.granularity = self.para.granularity
        self.num_train = 23967

        self.init_placeholder()  # init placeholder
        self.model()             # init prediction model


    def init_placeholder(self):
        '''
        :return:
        '''
        self.placeholders = {
            # 输入维度: [batch_size, input_len, num_nodes, num_features]
            'features': tf.placeholder(tf.float32, shape=[None, self.input_len, self.site_num, self.features],
                                       name='input_features'),

            # 输出标签: [batch_size, num_nodes, total_len, output_features]
            # 如果输出和输入特征维度一致，可以用同样的 self.features
            'labels': tf.placeholder(tf.float32, shape=[None, self.site_num, self.total_len, self.features],
                                     name='labels'),

            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout')
        }

    def adjecent(self):
        '''
        :return: adjacent matrix
        '''
        data = pd.read_csv(filepath_or_buffer=self.para.file_adj)
        adj = np.zeros(shape=[self.site_num, self.site_num])
        for line in data[['src_FID', 'nbr_FID']].values:
            adj[line[0]][line[1]] = 1
        return adj

    def model(self):
        '''
        :return:
        '''

        def TGCN(_X, labels,adj):
            ###
            # 假设 _X 形状为 (B, 12, 66, 33)，首先处理33维特征的每一维
            feature_1 = _X[:, :, :, 0:1]  # 取出第一维特征 (B, 12, 66, 1)
            feature_2 = _X[:, :, :, 1:]  # 取出后32维特征 (B, 12, 66, 32)

            # 第一维通过全连接层映射为32维
            feature_1_fc = tf.layers.dense(feature_1, units=32, activation=tf.nn.relu,
                                           name="feature_1_fc")  # (B, 12, 66, 32)

            # 拼接第一维的映射和后32维特征，得到64维特征
            _X = tf.concat([feature_1_fc, feature_2], axis=-1)  # (B, 12, 66, 64)
            cell_1 = tgcnCell(num_units=self.hidden_size, adj=adj, num_nodes=self.site_num)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)  # 可用多层
            _X = tf.unstack(_X, axis=1)
            outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
            print('outputs length is : ', len(outputs))
            print('outputs shape is : ', outputs[-1].shape)
            m = []
            for i in outputs:
                o = tf.reshape(i, shape=[-1, self.site_num, self.hidden_size])

                o = tf.reshape(o, shape=[-1, self.hidden_size])

                m.append(o)
            last_output = m[-1]

            print('last_output shape is : ', last_output.shape)
            last_output = tf.reshape(last_output, [-1, self.site_num, self.hidden_size])
            print('last_output shape is : ', last_output.shape)
            #last_output = tf.reshape(last_output, [-1, self.site_num, self.hidden_size])

            # ---------------- 调整标签维度 ----------------

            # ---------------- 调整标签维度 ----------------
            # labels: (batch, nodes, time, features) -> (batch, time, nodes, features)
            label_features = tf.transpose(labels, perm=[0, 2, 1, 3])
            label_features = label_features[:, -self.output_len:, :, 1:]  # 取最后 output_len 步，去掉第0维特征
            print('label_features shape after transpose and slice: ', label_features.shape)

            # 将标签特征reshape为 (batch, 66, 12*32)
            label_features_reshaped = tf.reshape(label_features,
                                                 shape=[-1, self.site_num, self.output_len * 32])  # (B, 66, 12*32)

            # 拼接 last_output 和 label_features_reshaped
            last_output = tf.concat([last_output, label_features_reshaped], axis=-1)  # (batch, nodes, 100 + 12*32)

            '''
            # labels: (batch, nodes, time, features) -> (batch, time, nodes, features)
            label_features = tf.transpose(labels, perm=[0, 2, 1, 3])
            label_features = label_features[:, -self.output_len:, :, 1:]  # 取最后 output_len 步，去掉第0维特征
            print('label_features shape after transpose and slice: ', label_features.shape)

            # 线性映射到100维
            label_features_mapped = tf.layers.dense(label_features, units=100, activation=tf.nn.relu,
                                                    name='label_dense')


            # 卷积整合时间维度
            conv_out = tf.layers.conv2d(label_features_mapped, filters=100, kernel_size=(self.output_len, 1),
                                        strides=(1, 1), padding='valid', name='label_conv')

            conv_out = tf.squeeze(conv_out, axis=1)  # 去掉时间维度 -> (batch, nodes, 100)

            last_output = last_output + conv_out  # (batch, nodes, 100)
'''

            last_output = tf.layers.dense(inputs=last_output, units=64, activation=tf.nn.relu, name='layer_1')
            output = tf.layers.dense(inputs=last_output, units=self.output_len, name='output_y')
            print('output shape is : ', output.shape)

            return output, m, states

        adj = self.adjecent()

        self.pre, _, _ = TGCN(self.placeholders['features'], self.placeholders['labels'], adj=adj)
        self.pre = self.pre * (self.std) + self.mean
        #print('pres shape is : ', self.pre.shape)

        print("self.placeholders['labels']",self.placeholders['labels'].shape)

        self.loss = mae_los(self.pre, self.placeholders['labels'][:,:,self.input_len:,0])
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def test(self):
        '''
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def initialize_session(self,session):
        self.sess = session
        self.saver = tf.train.Saver()

    def run_epoch(self, trainX, trainL, valX, valL,testX, testL):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''
        max_mae = 100
        wait = 0
        patience = 30  # 连续多少轮验证指标没提升就停止
        shape = trainX.shape
        num_batch = math.ceil(shape[0] / self.batch_size)
        self.num_train=shape[0]
        self.sess.run(tf.global_variables_initializer())
        start_time = datetime.datetime.now()
        iteration=1
        for epoch in range(self.epochs):
            # shuffle
            permutation = np.random.permutation(shape[0])
            trainX = trainX[permutation]
            trainL = trainL[permutation]
            for batch_idx in range(num_batch):
                iteration+=1
                start_idx = batch_idx * self.batch_size
                end_idx = min(shape[0], (batch_idx + 1) * self.batch_size)
                xs = trainX[start_idx : end_idx]

                labels = trainL[start_idx : end_idx]
                #print("labels:", labels.shape)
                #labels = labels[..., 0]
                #print("labels:",labels.shape)
                feed_dict = construct_feed_dict(features=xs,
                                                labels=labels,
                                                placeholders=self.placeholders)
                feed_dict.update({self.placeholders['dropout']: self.para.dropout})
                #print("labels:", labels.shape)

                loss_, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
                #print("labels:", labels.shape)

                if iteration == 500:
                    end_time = datetime.datetime.now()
                    total_time = end_time - start_time
                    print("Total running times is : %f" % total_time.total_seconds())

            print('validation')
            mae = self.evaluate(valX, valL)

            print("Epoch %d, Validation MAE: %.4f" % (epoch + 1, mae))
            test_mae = self.evaluate(testX, testL)
            print("Epoch %d, Validation MAE: %.4f" % (epoch + 1, mae))
            print("Epoch %d, Test MAE: %.4f" % (epoch + 1, test_mae))

            # 早停逻辑
            if mae < max_mae:
                max_mae = mae
                wait = 0
                # 保存最优模型
                self.saver.save(self.sess, save_path=self.para.save_path)
                print("Best model saved with MAE: %.4f" % max_mae)
            else:
                wait += 1
                print("No improvement in validation MAE. Wait: %d/%d" % (wait, patience))
                if wait >= patience:
                    print("Early stopping triggered. Training stopped.")
                    return

    #验证集


    def evaluate(self, testX, testL):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        labels_list, pres_list = list(), list()
        if not self.is_training:
            # model_file = tf.train.latest_checkpoint(self.para.save_path)
            saver = tf.train.import_meta_graph(self.para.save_path + '.meta')
            # saver.restore(sess, args.model_file)
            print('the model weights has been loaded:')
            saver.restore(self.sess, self.para.save_path)

        parameters = 0
        for variable in tf.trainable_variables():
            parameters += np.product([x.value for x in variable.get_shape()])
        print('trainable parameters: {:,}'.format(parameters))

        textX_shape = testX.shape
        total_batch = math.ceil(textX_shape[0] / self.batch_size)
        start_time = datetime.datetime.now()
        for b_idx in range(total_batch):
            start_idx = b_idx * self.batch_size
            end_idx = min(textX_shape[0], (b_idx + 1) * self.batch_size)
            xs = testX[start_idx: end_idx]
            labels = testL[start_idx: end_idx]
            #labels = labels[..., 0]
            feed_dict = construct_feed_dict(features=xs,
                                            labels=labels,
                                            placeholders=self.placeholders)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            pre = self.sess.run((self.pre), feed_dict=feed_dict)

            labels_list.append(labels[:,:,self.input_len:,0])
            pres_list.append(pre)

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

        labels_list = np.concatenate(labels_list, axis=0)
        pres_list = np.concatenate(pres_list, axis=0)
        np.savez_compressed('data/T-GCN-' + 'YINCHUAN', **{'prediction': pres_list, 'truth': labels_list})

        if not self.is_training:
            print('                MAE\t\tRMSE\t\tMAPE')
            for (l,r) in [(0,66)]:
                for i in range(self.output_len):
                    mae, rmse, mape = metric(pres_list[:,l:r,i], labels_list[:,l:r,i])
                    print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, mae, rmse, mape * 100))
                mae, rmse, mape = metric(pres_list[:,l:r], labels_list[:,l:r])  # 产生预测指标
                print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
                print('\n')
        mae, rmse, mape = metric(pres_list, labels_list)
        return mae

def main(argv=None):
    '''
    :param argv:
    :return:
    '''

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = InteractiveSession(config=config)
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:
        para.is_training = True
    else:
        para.batch_size = 1
        para.is_training = False

    trainX, trainDoW, trainM, trainL, trainXAll, valX, valDoW, valM, valL, valXAll, testX, testDoW, testM, testL, testXAll, mean, std = loadData(para)


    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainL.shape))
    print('valX:   %s\t\tvalY:   %s' % (valX.shape, valL.shape))
    print('testX:  %s\t\ttestY:  %s' % (testX.shape, testL.shape))
    print('data loaded!')

    pre_model = Model(para, mean, std)
    pre_model.initialize_session(session)
    if int(val) == 1:
        pre_model.run_epoch(trainX, trainL, valX, valL, testX, testL)
    else:
        pre_model.evaluate(testX, testL)

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()