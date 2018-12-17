import tensorflow as tf
import numpy as np


def conv2d(x, w, stride=2, padding='SAME'):
    return tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding)


def weight_var(shape, name='w_', init=tf.truncated_normal_initializer(stddev=0.02)):
    return tf.get_variable(name, shape, initializer=init)


def bias_var(shape, name='b_'):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def max_pool_layer(x,ksize=2,stride=2,padding='SAME',name='max_pool'):
    return tf.nn.max_pool(x,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding,name=name)

def fc_layer(x, w_shape, b_shape, keep_prob,activation=tf.nn.relu, batch_norm=None, dropout=False, name="liner", reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        w = weight_var(w_shape)
        b = bias_var(b_shape)
        h = tf.matmul(x, w) + b
        if activation:
            h = activation(h)
        if batch_norm is not None:
            h = batch_norm(h)
        if dropout:
            h = tf.nn.dropout(h, keep_prob=keep_prob)
        return h


def conv_layer(x, w_shape, b_shape, activation=tf.nn.relu, batch_norm=None, stride=2, name='conv2d', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        w = weight_var(w_shape, name='w_' + name)
        b = bias_var(b_shape, name='b_' + name)
        h = conv2d(x, w, stride=stride) + b
        if activation is not None:
            h = activation(h)
        return h


def classifier(x,keep_prob,name='CNN', image_size=32, channels=18, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        #mode2
        conv1 = conv_layer(x,[1,1,channels,36],[36],stride=1, name='conv_1', reuse=reuse)

        conv2 = conv_layer(conv1, [2, 2, 36, 72], [72], stride=1, name='conv_2', reuse=reuse)
        pool2 = max_pool_layer(conv2,stride=2,ksize=2,name='max_pool_2')

        conv3 = conv_layer(pool2, [2, 2, 72, 144], [144], stride=1, name='conv_3', reuse=reuse)
        pool3 = max_pool_layer(conv3, stride=2, ksize=2, name='max_pool_3')

        conv4 = conv_layer(pool3, [2, 2, 144, 288], [288], stride=1, name='conv_4', reuse=reuse)
        pool4 = max_pool_layer(conv4, stride=2, ksize=2, name='max_pool_4')

        pool4 = tf.reshape(pool4, [-1, 4*4*288])

        fc1 = fc_layer(pool4, [4*4*288, 2048], [2048], keep_prob,name='fc_1', reuse=reuse)
        fc2 = fc_layer(fc1, [2048, 1024], [1024], keep_prob,name='fc_2', reuse=reuse)
        fc3 = fc_layer(fc2, [1024, 512], [512], keep_prob,name='fc_3', reuse=reuse)
        fc4 = fc_layer(fc3, [int(fc3.get_shape()[-1]), 17], [17], keep_prob,activation=False, name='fc_4', reuse=reuse)

        return (fc4, tf.nn.softmax(fc4))

        '''
        #model 1
        conv1 = conv_layer(x, [4, 4, channels, 36], [36], stride=2, name='conv_1', reuse=reuse)
        conv2 = conv_layer(conv1, [4, 4, int(conv1.get_shape()[-1]), 36 * 2], [36 * 2], stride=2, name='conv_2',
                           reuse=reuse)
        conv3 = conv_layer(conv2, [4, 4, int(conv2.get_shape()[-1]), 36 * 2 * 2], [36 * 2 * 2], stride=2, name='conv_3',
                           reuse=reuse)

        conv3 = tf.reshape(conv3, [-1, 36*2*2*4*4])
        fc1 = fc_layer(conv3, [36*2*2*4*4, 1024], [1024], name='fc_1', reuse=reuse)
        fc2 = fc_layer(fc1, [int(fc1.get_shape()[-1]),512],[512], name='fc_2',reuse=reuse)
        fc3 = fc_layer(fc2, [int(fc2.get_shape()[-1]), 17], [17], activation=False, name='fc_3', reuse=reuse)

        return (fc3,tf.nn.softmax(fc3))
        '''

def get_accurancy(y_out, label_batch,name='accu'):
    with tf.variable_scope(name) as scope:
        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
