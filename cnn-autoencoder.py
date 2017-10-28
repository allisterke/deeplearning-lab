# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

batch_size = 32

class Discriminator:
    def __init__(self):
        with tf.variable_scope('discriminator'):
            self.W_conv1 = weight_variable([5, 5, 1, 1024])
            self.b_conv1 = bias_variable([1024])

            self.W_conv2 = weight_variable([5, 5, 1024, 1])
            self.b_conv2 = bias_variable([1])

            self.W_fc1 = weight_variable([49, 1024])
            self.b_fc1 = bias_variable([1024])

            self.W_fc2 = weight_variable([1024, 100])
            self.b_fc2 = bias_variable([100])

            self.W_fc3 = weight_variable([100, 1024])
            self.b_fc3 = bias_variable([1024])

            self.W_fc4 = weight_variable([1024, 1])
            self.b_fc4 = bias_variable([1])

    def embed(self, x):
        x = tf.reshape(x, [-1, 28, 28, 1])

        x = tf.nn.conv2d(x, self.W_conv1, [1, 2, 2, 1], "SAME") + self.b_conv1
        # x = tf.nn.relu(x)
        x = tf.nn.tanh(x)

        x = tf.nn.conv2d(x, self.W_conv2, [1, 2, 2, 1], "SAME") + self.b_conv2
        # x = tf.nn.relu(x)
        x = tf.nn.tanh(x)

        x = tf.reshape(x, [-1, 49])

        x = tf.matmul(x, self.W_fc1) + self.b_fc1
        # x = tf.nn.relu(x)
        x = tf.nn.tanh(x)

        x = tf.matmul(x, self.W_fc2) + self.b_fc2
        x = tf.nn.sigmoid(x)

        return x

    def discriminate(self, x):
        x = self.embed(x)

        x = tf.matmul(x, self.W_fc3) + self.b_fc3
        x = tf.nn.tanh(x)

        x = tf.matmul(x, self.W_fc4) + self.b_fc4
        x = tf.nn.sigmoid(x)

        return x

class Generator:
    def __init__(self):
        with tf.variable_scope('generator'):
            self.W_fc1 = weight_variable([100, 100])
            self.b_fc1 = bias_variable([100])

            self.W_fc2 = weight_variable([100, 7*7*1024])
            self.b_fc2 = bias_variable([7*7*1024])

            self.W_dconv0 = weight_variable([5, 5, 64, 1024])
            self.b_dconv0 = bias_variable([64])

            self.W_dconv1 = weight_variable([5, 5, 16, 64])
            self.b_dconv1 = bias_variable([16])

            self.W_dconv2 = weight_variable([5, 5, 1, 16])
            self.b_dconv2 = bias_variable([1])

    def generate(self, x):
        x = tf.matmul(x, self.W_fc1) + self.b_fc1
        # x = tf.nn.relu(x)
        x = tf.nn.tanh(x)

        x = tf.matmul(x, self.W_fc2) + self.b_fc2
        x = tf.reshape(x, [-1, 7, 7, 1024])
        # input = tf.nn.batch_normalization(input, 0, 0.1, 0, 1, 1e-5)
        # x = tf.nn.relu(x)
        x = tf.nn.tanh(x)

        x = tf.nn.conv2d_transpose(x, self.W_dconv0, [int(x.get_shape()[0]), 14, 14, 64], [1, 2, 2, 1], "SAME") + self.b_dconv0
        # h_dconv0 = tf.nn.batch_normalization(h_dconv0, 0, 0.1, 0, 1, 1e-5)
        # x = tf.nn.relu(x)
        x = tf.nn.tanh(x)

        x = tf.nn.conv2d_transpose(x, self.W_dconv1, [int(x.get_shape()[0]), 28, 28, 16], [1, 2, 2, 1], "SAME") + self.b_dconv1
        # h_dconv1 = tf.nn.batch_normalization(h_dconv1, 0, 0.1, 0, 1, 1e-5)
        # x = tf.nn.relu(x)
        x = tf.nn.tanh(x)

        x = tf.nn.conv2d_transpose(x, self.W_dconv2, [int(x.get_shape()[0]), 28, 28, 1], [1, 1, 1, 1], "SAME") + self.b_dconv2
        x = tf.nn.sigmoid(x)

        x = tf.reshape(x, (-1, 28*28))
        return x


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fake_embedding(n):
    return np.asarray(np.random.random_sample((n, 100)), np.float32)

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    discriminator = Discriminator()
    generator = Generator()

    rx = tf.placeholder(tf.float32, [batch_size, 784])
    ry = discriminator.discriminate(rx)
    re = discriminator.embed(rx)
    ro = generator.generate(re)

    ge = tf.placeholder(tf.float32, [batch_size, 100])
    go = generator.generate(ge)
    gy = discriminator.discriminate(go)

    error1 = \
        tf.reduce_mean(tf.square(ry - 1)) + \
        tf.reduce_mean(tf.square(gy))

    error2 = \
        tf.reduce_mean(tf.square(gy - 1))

    error3 = \
        tf.reduce_mean(tf.square(rx - ro))

    error1 = error2 = error3

    learning_rate = tf.placeholder(tf.float32)
    train_step1 = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(error1, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    train_step2 = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(error2, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))
    train_step3 = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(error3, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator"))
    # train_step1 = tf.train.AdamOptimizer(1e-4).minimize(error1)
    # train_step2 = tf.train.AdamOptimizer(1e-4).minimize(error2)

    with tf.Session() as sess:
        rate = 1e1
        sess.run(tf.global_variables_initializer())
        gene_batch = fake_embedding(batch_size)
        for i in range(20000):
        # for i in range(10):
            batch = mnist.train.next_batch(batch_size)
            real_batch = batch[0]
            if i % 10 == 0:
                rate /= 1.001
            if i % 1 == 0:
                train_error1, _ = \
                    sess.run([error1, train_step1],
                                     feed_dict={
                                         learning_rate: rate,
                                         rx: real_batch, ge: gene_batch
                                     })
                train_error2, _ = \
                    sess.run([error2, train_step2],
                                     feed_dict={
                                         learning_rate: rate,
                                         rx: real_batch, ge: gene_batch
                                     })
                train_error2, _ = \
                    sess.run([error2, train_step2],
                                     feed_dict={
                                         learning_rate: rate,
                                         rx: real_batch, ge: gene_batch
                                     })
                train_error3, _ = \
                    sess.run([error3, train_step3],
                                     feed_dict={
                                         learning_rate: rate,
                                         rx: real_batch, ge: gene_batch
                                     })
                print('step %d, d_loss: %g, g_loss: %g, s_loss: %g'
                      % (i, train_error1, train_error2, train_error3))

            if i % 10 == 0:
                for i in range(10):
                    image = real_batch[i] * 256
                    png = sess.run(tf.image.encode_png(np.reshape(image, [28, 28, 1])))
                    with open("%dr.png" % i, "wb") as f:
                        f.write(png)
                # generated = fake_embedding(100)
                # decoded = sess.run(tf.cast(generator.generate(generated) * 256, tf.uint8))
                _re, decoded = sess.run([re, ro * 256], feed_dict={rx: real_batch})
                for i in range(10):
                    image = decoded[i]
                    png = sess.run(tf.image.encode_png(tf.cast(tf.reshape(image, [28, 28, 1]), tf.uint8)))
                    with open("%d.png" % i, "wb") as f:
                        f.write(png)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                                            default='./mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
