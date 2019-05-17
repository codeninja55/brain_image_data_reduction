# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# brain_image_data_reduction
# autoencoder.py
# 
# Attributions: 
# [1] N. Shukla. Machine Learning with Tensorflow
# ----------------------------------------------------------------------------------------------------------------------

__author__ = 'Andrew Che <@codeninja55>'
__copyright__ = 'Copyright (C) 2019, Andrew Che <@codeninja55>'
__credits__ = ['']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = 'Andrew Che'
__email__ = 'andrew@neuraldev.io'
__status__ = '{dev_status}'
__date__ = '2019.05.14'

"""autoencoder.py: 

{Description}
"""
import tensorflow as tf
from IPython.display import clear_output


class Autoencoder:

    def __init__(self, input_dim, hidden_dim, epoch=250, learning_rate=0.001):
        # init variables
        self.epoch = epoch
        self.alpha = learning_rate
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])

        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.sigmoid(tf.matmul(x, weights) + biases)

        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([input_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases

        self.x = x
        self.encoded = encoded
        self.decoded = decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)
        self.saver = tf.train.Saver()

    def train(self, data):
        # train on a dataset
        n_samples = len(data)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.epoch):
                # clear_output()
                for j in range(n_samples):
                    l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: [data[j]]})

                if i % 10 == 0:
                    print('epoch {}: loss = {}'.format(i, l))
                    self.saver.save(sess, './model.ckpt')
            self.saver.save(sess, './model.ckpt')

    def test(self, data):
        # test on some new data
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data})
        print('input', data)
        print('compressed', hidden)
        print('reconstructed', reconstructed)
        return reconstructed
