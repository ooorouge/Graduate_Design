import tensorflow as tf
import numpy as np

conv_size = 3
pool_size = 3
stride_step = 3
NNN = 38
a = np.ones(shape=[4,12,12], dtype=np.float32)

a_4 = tf.expand_dims(a, 1)

filtera4 = tf.random_normal(shape=[1,3,12,16])
conva4 = tf.nn.conv2d(a_4, filter=filtera4, strides=[1, 1, 2, 1], padding='VALID')

with tf.Session() as sess:
    print(np.shape(sess.run(conva4)))
#a_reshape = tf.reshape(tf.convert_to_tensor(a), shape=[4, NNN])
#place = tf.placeholder(dtype=tf.float32, shape=[None, 9])
#place = tf.expand_dims(place, axis=2)
#
#hotlabel = [1, 2, 0, 3, 2]
#hotlabel = np.array(hotlabel)
#hotlabel = tf.one_hot(hotlabel, depth=4)
#
##weight = tf.random_normal(shape=[39, 1])
##bias = tf.random_normal(shape=[5, 1])
#
#filter = tf.random_normal(shape=[1,conv_size,1,1])
#filter2 = tf.random_normal(shape=[1,conv_size+2,1,1])
#filter_more_out = tf.random_normal(shape=[1,conv_size+2,2,5])
#pool = tf.random_normal(shape=[1,1,pool_size,1])
#
#a_3 = tf.expand_dims(a, 1)
#a_4 = tf.expand_dims(a_3, 3)
#
#c = tf.nn.conv2d(a_4, filter, strides=[1,1,stride_step,1], padding='VALID')
#c2 = tf.nn.conv2d(a_4, filter2, strides=[1,1,stride_step,1], padding='VALID')
#p = tf.nn.max_pool(a_4, ksize=[1,1,pool_size,1], strides=[1,1,stride_step,1], padding='VALID')
##concat_cp = tf.concat([c, p], axis=3)
##concat12 = tf.concat([c, c2], axis=3)
#
##conv4concat = tf.nn.conv2d(concat_cp, filter_more_out, strides=[1,1,stride_step+6,1], padding='VALID')
##conv4concat = tf.squeeze(conv4concat)
#tt = [1, 2, 3, -4]
#
##softmax4 = tf.nn.softmax(conv4concat, axis=1)
#
##c_weighted = tf.add(tf.matmul(c_squeeze, weight), bias)
#
#with tf.Session() as sess:
#    print('input=[4,{2}], c/k={0}, stride={1}'.format(conv_size, stride_step, NNN))
#    print(np.shape(sess.run(c)), end=' ')
#    print('is c with convsize={}, stride={}'.format(conv_size, stride_step))
#    print(np.shape(sess.run(c2)), end=' ')
#    print('is c2 with convsize={}'.format(conv_size+2))
#    print(np.shape(sess.run(p)), end=' ')
#    print('is c2 with poolsize={}'.format(pool_size))
#    #print(np.shape(sess.run(conv4concat)), end=' ')
#    print('argmax is {}'.format(np.argmax(tt)))
#    #print(np.shape(concat_cp), end=' ')
#    print('is softmax4')
#    #print(sess.run(a_4), end=' ')
#    #print('is a_4 with shape={}'.format(np.shape(a_4)))


