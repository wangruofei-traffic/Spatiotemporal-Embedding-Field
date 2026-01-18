# -- coding: utf-8 --
from __future__ import division
from __future__ import print_function
from models.hyparameter import parameter
from models.embedding import embedding
from models.inits import *
from models.utils import *
from models.data_load import *
from models.bi_lstm import BilstmClass
import datetime

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class Model(object):
    def __init__(self, para, min, max):
        self.para = para
        self.min = min
        self.max = max
        self.input_len = self.para.input_length
        self.output_len = self.para.output_length
        self.total_len = self.input_len + self.output_len
        self.features = self.para.features
        self.batch_size = self.para.batch_size
        self.epochs = self.para.epoch
        self.site_num = self.para.site_num
        self.hidden_layer = self.para.hidden_layer
        self.emb_size = self.para.emb_size
        self.is_training = self.para.is_training
        self.learning_rate = self.para.learning_rate
        self.model_name = self.para.model_name
        self.granularity = self.para.granularity
        self.decay_epoch=self.para.decay_epoch
        self.num_train = 23967

        # placeholders
        self.placeholders = {
            'position': tf.placeholder(tf.int32, shape=(1, self.para.site_num), name='input_position'),
            'week': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='day_of_week'),
            'day': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_day'),
            'hour': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_hour'),
            'minute': tf.placeholder(tf.int32, shape=(None, self.para.site_num), name='input_minute'),
            'features': tf.placeholder(tf.float32, shape=[None, self.input_len, self.site_num, self.features],name='inputs'),
            'features_all': tf.placeholder(tf.float32, shape=[None, self.input_len+self.output_len, self.site_num, self.features],name='total_inputs'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.site_num, self.total_len , self.features], name='labels'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'random_mask': tf.placeholder(tf.int32, shape=(None, self.site_num, self.total_len), name='mask'),
            'is_training': tf.placeholder(shape=(), dtype=tf.bool)
        }
        self.embeddings()
        self.model()

    def embeddings(self):
        '''
        :return:
        '''
        p_emd = embedding(self.placeholders['position'], vocab_size=self.site_num, num_units=self.para.emb_size, scale=False, scope="position_embed")
        p_emd = tf.reshape(p_emd, shape=[1, self.site_num, self.emb_size])
        self.p_emd = tf.expand_dims(p_emd, axis=0)

        w_emb = embedding(self.placeholders['week'], vocab_size=7, num_units=self.emb_size, scale=False,
                          scope="day_of_week_embed")
        self.w_emd = tf.reshape(w_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        d_emb = embedding(self.placeholders['day'], vocab_size=32, num_units=self.emb_size, scale=False, scope="day_embed")
        self.d_emd = tf.reshape(d_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        h_emb = embedding(self.placeholders['hour'], vocab_size=24, num_units=self.para.emb_size, scale=False, scope="hour_embed")
        self.h_emd = tf.reshape(h_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        m_emb = embedding(self.placeholders['minute'], vocab_size=24 * 60 //self.granularity, num_units=self.para.emb_size, scale=False, scope="minute_embed")
        self.m_emd = tf.reshape(m_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

    def model(self):
        '''
        :param batch_size: 64
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: True
        :return:
        '''

        features = tf.reshape(self.placeholders['features'], shape=[self.batch_size,
                                                                      self.input_len,
                                                                      self.site_num,
                                                                      self.features])
        # this step use to encoding the input series data
        encoder_init = BilstmClass(self.para, placeholders=self.placeholders)
        inputs = tf.transpose(features, perm=[0, 2, 1, 3])
        inputs = tf.reshape(inputs, shape=[self.batch_size * self.site_num,
                                           self.input_len,
                                           self.features])
        h_states = encoder_init.encoding(inputs)

        labels = tf.reshape(self.placeholders['labels'], shape=[self.batch_size,
                                                                    self.site_num, self.total_len , self.features])



        Y_attr = tf.reshape(labels, shape=[self.batch_size * self.site_num,
                                           self.total_len , self.features])  # 后 32 维
        Y_attr = Y_attr[:, -12:, 1:]


        # decoder
        print('#................................in the decoder step......................................#')
        # this step to presict the polutant concentration
        pre = encoder_init.decoding(h_states, self.site_num, future_attr=Y_attr)
        print('pres shape is : ', pre.shape)

        self.pre = pre * (self.max) + self.min
        observed = self.placeholders['labels']
        predicted = self.pre
        self.loss = mae_los(predicted, observed[:, :, self.input_len:,0])
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def test(self):
        '''
        :param batch_size: usually use 1
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: False
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def initialize_session(self,session):
        self.sess = session
        self.saver = tf.train.Saver()

    def re_current(self, a, max, min):
        return a * (max - min) + min
        # return [num * (max - min) + min for num in a]

    def run_epoch(self,trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''
        max_mae = 100
        patience = 20  # 允许连续多少轮没有提升
        wait = 0
        shape = trainX.shape
        num_batch = math.floor(shape[0] / self.batch_size)
        self.num_train=shape[0]
        self.sess.run(tf.global_variables_initializer())
        iteration = 0
        start_time = datetime.datetime.now()
        for epoch in range(self.epochs):
            # shuffle
            permutation = np.random.permutation(shape[0])
            trainX = trainX[permutation]
            trainDoW = trainDoW[permutation]
            trainD = trainD[permutation]
            trainH = trainH[permutation]
            trainM = trainM[permutation]
            trainL = trainL[permutation]
            trainXAll = trainXAll[permutation]
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(shape[0], (batch_idx + 1) * self.batch_size)
                random_mask = np.random.randint(low=0,high=2,size=[self.batch_size, self.site_num, self.total_len],dtype=int)
                xs = trainX[start_idx : end_idx]
                d_of_week = np.reshape(trainDoW[start_idx : end_idx], [-1, self.site_num])
                day = np.reshape(trainD[start_idx : end_idx], [-1, self.site_num])
                hour = np.reshape(trainH[start_idx : end_idx], [-1, self.site_num])
                minute = np.reshape(trainM[start_idx : end_idx], [-1, self.site_num])
                labels = trainL[start_idx : end_idx]
                xs_all = trainXAll[start_idx : end_idx]
                feed_dict = construct_feed_dict(xs, xs_all, labels, d_of_week, day, hour, minute, mask=random_mask, placeholders=self.placeholders, site=self.site_num)
                feed_dict.update({self.placeholders['dropout']: self.para.dropout})
                l,_ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
                # print("after %d steps and %d epochs, the training total average loss is : %.6f" % (batch_idx, epoch+1, l))

                iteration+=1
                if iteration == 100:
                    end_time = datetime.datetime.now()
                    total_time = end_time - start_time
                    print("Total running times is : %f" % total_time.total_seconds())

                # if batch_idx % 100 == 0:
            print('validation')
            mae = self.evaluate(valX, valDoW, valD, valH, valM, valL, valXAll) # validate processing
            test_mae = self.evaluate(testX, testDoW, testD, testH, testM, testL, testXAll)  # validate processing
            print("test_mae: %f" % test_mae)
            if mae < max_mae:

                wait = 0
                self.saver.save(self.sess, save_path=self.para.save_path)
                print('Val loss decrease from {:.3f} to {:.3f}, '
                      'saving to {}, in epoch {:d}'.format(max_mae, mae, self.para.save_path, epoch))
                max_mae = mae
            else:
                wait += 1
                print("Epoch %d: validation MAE did not improve. Wait %d/%d" % (epoch + 1, wait, patience))

            if wait >= patience:
                print("Early stopping triggered. No improvement for %d epochs." % patience)
                break



    def evaluate(self, testX, testDoW, testD, testH, testM, testL, testXAll):
        '''
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

        start_time = datetime.datetime.now()
        textX_shape = testX.shape
        total_batch = math.floor(textX_shape[0] / self.batch_size)
        for b_idx in range(total_batch):
            start_idx = b_idx * self.batch_size
            end_idx = min(textX_shape[0], (b_idx + 1) * self.batch_size)
            random_mask = np.ones(shape=[self.batch_size,self.site_num,self.total_len],dtype=int)
            xs = testX[start_idx: end_idx]
            d_of_week = np.reshape(testDoW[start_idx: end_idx], [-1, self.site_num])
            day = np.reshape(testD[start_idx: end_idx], [-1, self.site_num])
            hour = np.reshape(testH[start_idx: end_idx], [-1, self.site_num])
            minute = np.reshape(testM[start_idx: end_idx], [-1, self.site_num])
            labels = testL[start_idx: end_idx]
            xs_all = testXAll[start_idx : end_idx]
            feed_dict = construct_feed_dict(xs, xs_all, labels, d_of_week, day, hour, minute, mask=random_mask, placeholders=self.placeholders,site=self.site_num, is_training=False)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            pre_s = self.sess.run((self.pre), feed_dict=feed_dict)

            labels_list.append(labels[:, :, self.input_len:,0])
            pres_list.append(pre_s)

        labels_list = np.concatenate(labels_list, axis=0)
        pres_list = np.concatenate(pres_list, axis=0)
        np.savez_compressed('data/LSTM_BILSTM-' + 'YINCHUAN', **{'prediction': pres_list, 'truth': labels_list})

        #print('                MAE\t\tRMSE\t\tMAPE')
        '''
        for (l,r) in [(0,13),(13,26),(26,66)]:
            for i in range(self.para.output_length):
                mae, rmse, mape = metric(pres_list[:,l:r,i], labels_list[:,l:r,i])
                print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, mae, rmse, mape * 100))
            mae, rmse, mape = metric(pres_list[:,l:r], labels_list[:,l:r])  # 产生预测指标
            print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
            print('\n')
        '''
        if not self.is_training:
            print('                MAE\t\tRMSE\t\tMAPE')
            for (l, r) in [(0, 170)]:
                for i in range(self.output_len):
                    mae, rmse, mape = metric(pres_list[:, l:r, i], labels_list[:, l:r, i])
                    print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, mae, rmse, mape * 100))
                mae, rmse, mape = metric(pres_list[:, l:r], labels_list[:, l:r])  # 产生预测指标
                print('average:         %.3f\t\t%.3f\t\t%.3f%%' % (mae, rmse, mape * 100))
                print('\n')
        mae, rmse, mape = metric(pres_list, labels_list)


        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

        # describe(label_list, predict_list)   #预测值可视化
        return mae


def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    config = ConfigProto()
    config.gpu_options.allow_growth = True
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

    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll, min, max = loadData(para)
    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainL.shape))
    print('valX:   %s\t\tvalY:   %s' % (valX.shape, valL.shape))
    print('testX:  %s\t\ttestY:  %s' % (testX.shape, testL.shape))
    print('data loaded!')
    pre_model = Model(para, min, max)
    pre_model.initialize_session(session)
    if int(val) == 1:
        pre_model.run_epoch(trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll)
    else:
        pre_model.evaluate(testX, testDoW, testD, testH, testM, testL, testXAll)

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()