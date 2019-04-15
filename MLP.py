import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

from IO2csv import *

import Globalparameters as glb


# -------load files and extract desired data and features-------
x_vals = np.zeros((None, 43),dtype=np.float)
x_vals_squeeze = np.zeros((None,42),dtype=np.float)
y_vals_squeeze = np.zeros(None,dtype=np.float)
x_vals, x_vals_squeeze, y_vals_squeeze = ReadFromCsv()
# Reset the graph for new run
ops.reset_default_graph()

# Create graph session
sess = tf.Session()
# Saver = tf.train.Saver()

# Set batch size for training
batch_size = 50
conv_size = 5
pool_size = 5
stride_size = 1

# Set random seed to make results reproducible
seed = 709
np.random.seed(seed)
tf.set_random_seed(seed)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals_squeeze[train_indices]
x_vals_test = x_vals_squeeze[test_indices]
y_vals_train = y_vals_squeeze[train_indices]
y_vals_test = y_vals_squeeze[test_indices]


# Define Variable Functions (weights and bias)
def init_weight(shape, st_dev, name):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev), name=name)
    return weight


def init_bias(shape, st_dev, name):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev), name=name)
    return bias


# Create Placeholders
x_data = tf.placeholder(shape=[glb.batch_size, 42], dtype=tf.float32)
y_target = tf.placeholder(shape=[glb.batch_size, 1], dtype=tf.float32)


# Create a fully connected layer:
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.sigmoid(layer)


# -------Create the first layer (50 hidden nodes)--------
weight_1 = init_weight(shape=[42, 25], st_dev=10.0, name='w1')
bias_1 = init_bias(shape=[25], st_dev=10.0, name='b1')
layer_1 = fully_connected(x_data, weight_1, bias_1)

# -------Create second layer (25 hidden nodes)--------
weight_2 = init_weight(shape=[21*3, 14], st_dev=10.0, name='w2')
bias_2 = init_bias(shape=[14], st_dev=10.0, name='b2')
layer_2 = fully_connected(layer_1, weight_2, bias_2)

# -------Create third layer (5 hidden nodes)--------
weight_3 = init_weight(shape=[6, 3], st_dev=10.0, name='w3')
bias_3 = init_bias(shape=[3], st_dev=10.0, name='b3')
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# -------Create activiation layer (sigmoid)---------
def ActivationLayer(input):
    return tf.nn.sigmoid(input)
layer_activation_output = ActivationLayer(layer_3)

# -------Create output layer (1 output value)--------
weight_4 = init_weight(shape=[3, 1], st_dev=10.0, name='w4')
bias_4 = init_bias(shape=[1], st_dev=10.0, name='b4')
final_output = fully_connected(layer_activation_output, weight_4, bias_4)

# Declare loss function (L2)
loss = tf.reduce_mean((y_target - final_output)**2)

# Declare optimizer
my_opt = tf.train.AdamOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)
# Saver.restore(sess, 'model_1_17')

# Training loop
loss_vec = []
test_loss = []
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    #if (i + 1) % 25 == 0:
    #    print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

# Model Accuracy
actuals = np.array([x[42] for x in x_vals])
test_actuals = actuals[test_indices]
train_actuals = actuals[train_indices]
test_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_test})]
train_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_train})]
#saver.save(sess, 'model_1_17')

## Results output for visualization
#ResultsOutput(test_preds, test_actuals)

test_preds = np.array([0.0 if x < 0.7 else 1.0 for x in test_preds])
train_preds = np.array([0.0 if x < 0.7 else 1.0 for x in train_preds])
#
merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起
writer = tf.summary.FileWriter('logs',sess.graph) #将训练日志写入到logs文件夹下
#
## Print out accuracies
#test_acc = np.mean([x == y for x, y in zip(test_preds, test_actuals)])
#train_acc = np.mean([x == y for x, y in zip(train_preds, train_actuals)])
#print('On predicting the category of low possibility from regression output (<60%):')
#print('Test Accuracy: {}'.format(test_acc))
#print('Train Accuracy: {}'.format(train_acc))
