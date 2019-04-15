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
batch_size = 5
whole_size = 200


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


class Training_Config1(object):
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


# ------------ Prequiste:                                 ------------ #
# ------------           make slices and init placeholder ------------ #
x_vals, x_vals_squeeze, y_vals_squeeze = fromcsv(length=123, size=whole_size)
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals_squeeze[train_indices]
x_vals_test = x_vals_squeeze[test_indices]
y_vals_train = y_vals_squeeze[train_indices]
y_vals_test = y_vals_squeeze[test_indices]
# ACCNN里的层都写了squeeze
acc331 = ACCNN(Training_Config1(3, 3, 1))
acc333 = ACCNN(Training_Config1(3, 3, 3))
acc553 = ACCNN(Training_Config1(5, 5, 3))
acc332 = ACCNN(Training_Config1(3, 3, 2))


def main(_):
    x_data = tf.placeholder(shape=[None, 120], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    # 这个地方可能报错 作用是先添加一个channel [N,120] -> [N, 120, 1]
    x_data_3 = tf.expand_dims(x_data, axis=2)
    # ------------ Model to build:                            ------------ #
    # ------------           make multiple layers             ------------ #
    # input data shape=[batch, 120, 1]
    # 4-Cs output 
    convolution_1 = acc331.conv_layer(x_data_3, 1, 4)
    maxpool_1 = acc331.pool_layer(convolution_1, type='MP')

    # Create shape adjust layer
    # input data shape=[batch, 120-3+1 / 1 = 118, 4]
    Incep1 = acc553.conv_layer(maxpool_1, 4, 4)

    # Create to reduce scale
    # input data shape=[batch, 38, 4]
    conv2_33 = acc333.conv_layer(Incep1, 4, 4)
    conv2_53 = acc553.conv_layer(Incep1, 4, 4)
    pool2_33 = acc333.pool_layer(Incep1, type='MP')
    Incep2 = tf.concat([conv2_33, conv2_53, pool2_33], axis=2)

    # Create to reduce for final avgpool
    # input data shape=[batch, 12, 12]
    Incep3 = acc332.conv_layer(Incep2, 12, 16)
    Incep3 = tf.expand_dims(Incep3, axis=1)
    # input data shape=[batch, 1, 5, 16]
    I3pool = tf.nn.avg_pool(Incep3, ksize=[1, 1, 5, 1], strides=[1,1,1,1], padding='VALID')
    I3pool = tf.squeeze(I3pool)

    # After a MLP output
    # input data shape=[batch, channels=16]
    layer1 = tf.layers.dense(inputs=tf.reshape(I3pool, shape=[-1, 16]), units=8, activation=tf.nn.sigmoid, trainable=True)
    logits = tf.layers.dense(inputs=tf.reshape(layer1, shape=[-1, 8]), units=3, activation=tf.nn.sigmoid, trainable=True)
    logits_softmax = tf.nn.softmax(logits)

    # loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_target, logits=logits_softmax)
    loss = tf.reduce_sum(loss)
    train_step = tf.train.AdamOptimizer(0.0006).minimize(loss)

    # Initialize Variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training loop
    loss_vec = []
    test_loss = []
    for i in range(100):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = y_vals_train[rand_index]
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

    # Model Accuracy
    test_actuals = [np.argmax(x) for x in y_vals_squeeze[test_indices]]
    train_actuals = [np.argmax(x) for x in y_vals_squeeze[train_indices]]
    test_preds = [np.argmax(x) for x in sess.run(logits_softmax, feed_dict={x_data: x_vals_test})]
    train_preds = [np.argmax(x) for x in sess.run(logits_softmax, feed_dict={x_data: x_vals_train})]

    # Print out accuracies
    test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)])
    train_acc = np.mean([x == y for x, y in zip(train_preds, train_actuals)])

    for i in range(len(test_actuals)):
        print(test_actuals[i], end=' and preds is ')
        print(test_preds[i])
    print('total test acc = {}%'.format(test_acc * 100))
    print('total train acc = {}%'.format(train_acc * 100))
    tv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    plt.plot(loss_vec)
    plt.show()


if __name__ == '__main__':
    tf.app.run()
