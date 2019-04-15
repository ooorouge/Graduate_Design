import tensorflow as tf
import numpy as np
import csv
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

ops.reset_default_graph()
sess = tf.Session()

# Ensure reproducibility
# 710 test 87 train 92
seed = 710
np.random.seed(seed)
tf.set_random_seed(seed)
whole_size = 200
lr = 0.0006
bs = 6
ts = 100
divide = 0.6


def fromcsv(length, size, filename="dataset_one_hot.csv"):
    """
    :param filename: filename and path
    :return:
            x_vals: Data with features and labels
            x_vals_squeeze: x_vals without labels
            y_vals_squeeze: only labels sorted
    """
    x_vals = np.zeros((size, length), dtype=np.float32)
    x_vals_squeeze = np.zeros((size, length - 3), dtype=np.float32)
    y_vals_squeeze = np.zeros((size, 3), dtype=np.float32)
    with open(filename, 'r', newline='') as f:
        csv_files = csv.reader(f)
        for i, line in enumerate(csv_files):
            if i >= size:
                break
            x_vals[i] = np.array(line)

    for i in range(size):
        y_vals_squeeze[i] = x_vals[i][-3:]
        x_vals_squeeze[i] = x_vals[i][:length - 3]

    return x_vals, x_vals_squeeze, y_vals_squeeze


class TrainingConfig1(object):
    """Parameters for training model"""

    def __init__(self, c, m, s):
        self.conv_size = c
        self.maxpool_size = m
        self.stride_size = s


class ACCNN(object):
    """docstring for ACCNN"""

    def __init__(self, config):
        self._ksize = config.conv_size
        self._maxpool_size = config.maxpool_size
        self._stride_size = config.stride_size

    def conv_layer(self, Rinput, inChannels, outChannels):
        Rinput = tf.expand_dims(Rinput, axis=1)
        with tf.name_scope('conv') as scope:
            filter_1 = tf.Variable(tf.random_normal(shape=[1, self._ksize, inChannels, outChannels]), name='{}{}{}'.format(
                self._ksize, self._maxpool_size, self._stride_size
            ))

        _conv = tf.nn.conv2d(Rinput,
                             filter=filter_1,
                             strides=[1, 1, self._stride_size, 1],
                             padding='VALID')

        _conv = tf.squeeze(_conv)
        return _conv

    def pool_layer(self, Rinput, type='MP'):
        Rinput = tf.expand_dims(Rinput, axis=1)

        if type is 'MP':
            _poolout = tf.nn.max_pool(Rinput,
                                      ksize=[1, 1, self._maxpool_size, 1],
                                      strides=[1, 1, self._stride_size, 1],
                                      padding='VALID')
            _poolout = tf.squeeze(_poolout)
            return _poolout

        if type is 'AVG':
            _poolout = tf.nn.avg_pool(Rinput,
                                      ksize=[1, 1, self._maxpool_size, 1],
                                      strides=[1, 1, self._stride_size, 1],
                                      padding='VALID')
            _poolout = tf.squeeze(_poolout)
            return _poolout


class AcMethods(object):
    def __init__(self, learning_rate=0.0006, batch_size=5, train_steps=200):
        self._lr = learning_rate
        self._bs = batch_size
        self._trainsteps = train_steps
        self.acc331 = ACCNN(TrainingConfig1(3, 3, 1))
        self.acc333 = ACCNN(TrainingConfig1(3, 3, 3))
        self.acc553 = ACCNN(TrainingConfig1(5, 5, 3))
        self.acc332 = ACCNN(TrainingConfig1(3, 3, 2))
        self.cnnloss_vec = []
        self.cnnacctrain = 0
        self.cnnacctest = 0
        self.mlploss_vec = []
        self.mlpacctrain = 0
        self.mlpacctest = 0

    def cnn(self, x_vals_train, x_vals_test, y_vals_train, y_vals_test):
        x_data = tf.placeholder(shape=[None, 120], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        # 这个地方可能报错 作用是先添加一个channel [N,120] -> [N, 120, 1]
        x_data_3 = tf.expand_dims(x_data, axis=2)
        # ------------ Model to build:                            ------------ #
        # ------------           make multiple layers             ------------ #
        # input data shape=[batch, 120, 1]
        # 4-Cs output
        convolution_1 = self.acc331.conv_layer(x_data_3, 1, 4)
        maxpool_1 = self.acc331.pool_layer(convolution_1, type='MP')

        # Create shape adjust layer
        # input data shape=[batch, 120-3+1 / 1 = 118, 4]
        Incep1 = self.acc553.conv_layer(maxpool_1, 4, 4)

        # Create to reduce scale
        # input data shape=[batch, 38, 4]
        conv2_33 = self.acc333.conv_layer(Incep1, 4, 4)
        conv2_53 = self.acc553.conv_layer(Incep1, 4, 4)
        pool2_33 = self.acc333.pool_layer(Incep1, type='MP')
        Incep2 = tf.concat([conv2_33, conv2_53, pool2_33], axis=2)

        # Create to reduce for final avgpool
        # input data shape=[batch, 12, 12]
        Incep3 = self.acc332.conv_layer(Incep2, 12, 16)
        Incep3 = tf.expand_dims(Incep3, axis=1)
        # input data shape=[batch, 1, 5, 16]
        I3pool = tf.nn.avg_pool(Incep3, ksize=[1, 1, 5, 1], strides=[
                                1, 1, 1, 1], padding='VALID')
        I3pool = tf.squeeze(I3pool)

        # After a MLP output
        # input data shape=[batch, channels=16]
        layer1 = tf.layers.dense(inputs=tf.reshape(I3pool, shape=[-1, 16]), units=8, activation=tf.nn.sigmoid,
                                 trainable=True)
        logits = tf.layers.dense(inputs=tf.reshape(layer1, shape=[-1, 8]), units=3, activation=tf.nn.sigmoid,
                                 trainable=True)
        logits_softmax = tf.nn.softmax(logits)

        # loss and optimizer
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_target, logits=logits_softmax)
        loss = tf.reduce_sum(loss)
        train_step = tf.train.AdamOptimizer(self._lr).minimize(loss)

        # Initialize Variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training loop
        test_loss = []
        for i in range(self._trainsteps):
            rand_index = np.random.choice(len(x_vals_train), size=self._bs)
            rand_x = x_vals_train[rand_index]
            rand_y = y_vals_train[rand_index]
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

            temp_loss = sess.run(
                loss, feed_dict={x_data: rand_x, y_target: rand_y})
            self.cnnloss_vec.append(temp_loss)

        # Model Accuracy
        test_actuals = [np.argmax(x) for x in y_vals_squeeze[test_indices]]
        train_actuals = [np.argmax(x) for x in y_vals_squeeze[train_indices]]
        test_preds = [np.argmax(x) for x in sess.run(
            logits_softmax, feed_dict={x_data: x_vals_test})]
        train_preds = [np.argmax(x) for x in sess.run(
            logits_softmax, feed_dict={x_data: x_vals_train})]

        # Print out accuracies
        test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)])
        train_acc = np.mean(
            [x == y for x, y in zip(train_preds, train_actuals)])
        self.cnnacctest = round(test_acc*100, 2)
        self.cnnacctrain = round(train_acc*100, 2)
        # for i in range(len(test_actuals)):
        #     print(test_actuals[i], end=' and preds is ')
        #     print(test_preds[i])
        print('cnn total test acc = {}%'.format(self.cnnacctest))
        print('cnn total train acc = {}%'.format(self.cnnacctrain))
        # tv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def mlp(self, x_vals_train, x_vals_test, y_vals_train, y_vals_test):
        x_data = tf.placeholder(shape=[None, 120], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)

        with tf.name_scope("mlp") as scope:
            layer1 = tf.layers.dense(inputs=tf.reshape(x_data, shape=[-1, 120]), units=240, activation=tf.nn.sigmoid,
                                     trainable=True)
            layer2 = tf.layers.dense(inputs=tf.reshape(layer1, shape=[-1, 240]), units=84, activation=tf.nn.relu,
                                     trainable=True)
            layer3 = tf.layers.dense(inputs=tf.reshape(layer2, shape=[-1, 84]), units=64, activation=tf.nn.relu,
                                     trainable=True)
            layer4 = tf.layers.dense(inputs=tf.reshape(layer3, shape=[-1, 64]), units=48, activation=tf.nn.relu,
                                     trainable=True)
            layer5 = tf.layers.dense(inputs=tf.reshape(layer4, shape=[-1, 48]), units=16, activation=tf.nn.relu,
                                     trainable=True)
            layer6 = tf.layers.dense(inputs=tf.reshape(layer5, shape=[-1, 16]), units=8, activation=tf.nn.relu,
                                     trainable=True)
            layer7 = tf.layers.dense(inputs=tf.reshape(layer6, shape=[-1, 8]), units=3, activation=tf.nn.sigmoid,
                                     trainable=True)

            # loss and optimizer
            logits_softmax = tf.nn.softmax(layer7)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_target, logits=logits_softmax)
            loss = tf.reduce_sum(loss)
            train_step = tf.train.AdamOptimizer(self._lr).minimize(loss)

            # Initialize Variables
            init = tf.global_variables_initializer()
            sess.run(init)

            # Training loop
            test_loss = []
            for i in range(self._trainsteps):
                rand_index = np.random.choice(len(x_vals_train), size=self._bs)
                rand_x = x_vals_train[rand_index]
                rand_y = y_vals_train[rand_index]
                sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

                temp_loss = sess.run(
                    loss, feed_dict={x_data: rand_x, y_target: rand_y})
                self.mlploss_vec.append(temp_loss)

            # Model Accuracy
            test_actuals = [np.argmax(x) for x in y_vals_squeeze[test_indices]]
            train_actuals = [np.argmax(x) for x in y_vals_squeeze[train_indices]]
            test_preds = [np.argmax(x) for x in sess.run(
                logits_softmax, feed_dict={x_data: x_vals_test})]
            train_preds = [np.argmax(x) for x in sess.run(
                logits_softmax, feed_dict={x_data: x_vals_train})]

            # Print out accuracies
            test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)])
            train_acc = np.mean(
                [x == y for x, y in zip(train_preds, train_actuals)])
            self.mlpacctest = round(test_acc*100, 2)
            self.mlpacctrain = round(train_acc*100, 2)
            # for i in range(len(test_actuals)):
            #     print(test_actuals[i], end=' and preds is ')
            #     print(test_preds[i])
            print('mlp total test acc = {}%'.format(self.mlpacctest))
            print('mlp total train acc = {}%'.format(self.mlpacctrain))
            # tv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


# ------------ Pre requisite:                                 ------------ #
# ------------           make slices and init placeholder     ------------ #
x_vals, x_vals_squeeze, y_vals_squeeze = fromcsv(length=123, size=whole_size)
train_indices = np.random.choice(
    len(x_vals), round(len(x_vals) * divide), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals_squeeze[train_indices]
x_vals_test = x_vals_squeeze[test_indices]
y_vals_train = y_vals_squeeze[train_indices]
y_vals_test = y_vals_squeeze[test_indices]


def main(_):
    method1 = AcMethods(learning_rate=lr, batch_size=bs, train_steps=ts)
    if isCNN:
        method1.cnn(x_vals_train, x_vals_test, y_vals_train, y_vals_test)
        str1 = 'CNN results-> test: {0}% train: {1}% \n'.format(str(method1.cnnacctest).ljust(7),
                                                                str(method1.cnnacctrain).ljust(7))
    else:
        method1.mlp(x_vals_train, x_vals_test, y_vals_train, y_vals_test)
        str2 = 'MLP results-> test: {0}% train: {1}% \n'.format(str(method1.mlpacctest).ljust(7),
                                                                str(method1.mlpacctrain).ljust(7))
    _l = 'lr= {} || bsize= {} || steps= {} || train:whole= {} || '.format(str(lr).ljust(8),
                                                                          str(bs).ljust(3),
                                                                          str(ts).ljust(5),
                                                                          str(divide).ljust(5)
                                                                          )
    with open('logs.txt', mode='a') as f:
        f.writelines(_l)
        if isCNN:
            f.writelines(str1)
        else:
            f.writelines(str2)


if __name__ == '__main__':
    isCNN = False
    tf.app.run()
