import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import csv

from data_generate import *
import Globalparameters

ops.reset_default_graph()

#----------1D data with sliding windows----------#
# Create graph session
sess = tf.Session()

# parameters for the run
data_size = 2*Globalparameters.scale
conv_size = 5
maxpool_size = 5
stride_size = 1
deadtime_deg = Globalparameters.deadtime_deg
deadtime_arc = Globalparameters.deadtime_arc
buoyant_for_test = Globalparameters.buoyant_for_test

# ensure reproducibility
seed = 709
np.random.seed(seed)
tf.set_random_seed(seed)

# Placeholder
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])


# --------Convolution--------
def conv_layer_1d(input_1d, input_filter, stride):
    """
    :param input_1d: 1D input array.
    :param input_filter: Filter to convolve across the input_1d array.
    :param stride: stride for filter.
    :return: array.
    """
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    convolution_output = tf.nn.conv2d(input_4d,
                                      filter=input_filter,
                                      strides=[1, 1, stride, 1],
                                      padding="VALID")
    # Get rid of extra dimensions
    conv_output_1d = tf.squeeze(convolution_output)
    return conv_output_1d

# Create filter for convolution.
my_filter = tf.Variable(tf.random_normal(shape=[1, conv_size, 1, 1]))
# Create convolution layer
my_convolution_output = conv_layer_1d(x_input_1d, my_filter, stride=stride_size)

# --------Activation--------
def activation(input_1d):
    return tf.nn.sigmoid(input_1d)

# Create activation layer
my_activation_output = activation(my_convolution_output)


# --------Max Pool--------
def max_pool(input_1d, width, stride):
    """
    :param input_1d: Input array to perform max-pool on.
    :param width: Width of 1d-window for max-pool
    :param stride: Stride of window across input array
    :return: max-pooled array
    """
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
                                 strides=[1, 1, stride, 1],
                                 padding='VALID')
    # Get rid of extra dimensions
    pool_output_1d = tf.squeeze(pool_output)
    return pool_output_1d

my_maxpool_output = max_pool(my_activation_output, width=maxpool_size, stride=stride_size)


# --------Fully Connected--------
def fully_connected(input_layer, num_outputs):

    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    # Initialize such weight
    weight = tf.random_normal(weight_shape, stddev=0.1)
    # Initialize the bias
    bias = tf.random_normal(shape=[num_outputs])
    # Make the 1D input array into a 2D array for matrix multiplication
    input_layer_2d = tf.expand_dims(input_layer, 0)
    # Perform the matrix multiplication and add the bias
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
    # Get rid of extra dimensions
    full_output_1d = tf.squeeze(full_output)
    return full_output_1d

my_full_output = fully_connected(my_maxpool_output, 8)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Save features to CSV format
open_csv = open('input_data_withSliding.csv', 'a', newline='')
csv_write = csv.writer(open_csv, dialect='excel')

for i in range(100):
    data_raw = DataSamples_Type2(Globalparameters.scale, 0*math.pi/180 + i / 100 * deadtime_arc, buoyant_for_test)
    data_1d = data_raw.extendData()
    data_1d = np.array(data_1d)
    feed_dict = {x_input_1d: data_1d}
    _ = sess.run(my_maxpool_output, feed_dict=feed_dict).tolist()
    _.append(0)
    csv_write.writerow(_)