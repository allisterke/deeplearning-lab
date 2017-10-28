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

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import math

FLAGS = None

batch_size = 10240

class DeepModel:
    def __init__(self, scope):
        with tf.variable_scope(scope):
            self.W_fc1 = weight_variable([1, 1024])
            self.b_fc1 = bias_variable([1024])

            self.W_fc2 = weight_variable([1024, 100])
            self.b_fc2 = bias_variable([100])

            self.W_fc3 = weight_variable([100, 1024])
            self.b_fc3 = bias_variable([1024])

            self.W_fc4 = weight_variable([1024, 1])
            self.b_fc4 = bias_variable([1])

    def build(self, x):
        x = tf.matmul(x, self.W_fc1) + self.b_fc1
        x = tf.nn.tanh(x)

        x = tf.matmul(x, self.W_fc2) + self.b_fc2
        x = tf.nn.tanh(x)

        x = tf.matmul(x, self.W_fc3) + self.b_fc3
        x = tf.nn.tanh(x)

        x = tf.matmul(x, self.W_fc4) + self.b_fc4
        x = tf.nn.sigmoid(x)

        return x

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def samples(n):
    return np.asarray(np.random.normal(size=(n, 1)), np.float32)

def fake_samples(n):
    return np.asarray(np.random.random_sample((n, 1)), np.float32)

def main(_):
    d_scope = 'd'
    g_scope = 'g'
    discriminator = DeepModel(d_scope)
    generator = DeepModel(g_scope)

    rx = tf.placeholder(tf.float32, [batch_size, 1])
    ry = discriminator.build(rx)

    ge = tf.placeholder(tf.float32, [batch_size, 1])
    go = generator.build(ge)
    go = tf.tan((go - 0.5) * np.pi)  # map range (0, 1) -> (-infinite, +infinite)
    gy = discriminator.build(go)

    error1 = \
        tf.reduce_mean(tf.square(ry - 1)) + \
        tf.reduce_mean(tf.square(gy))

    error2 = \
        tf.reduce_mean(tf.square(gy - 1))

    learning_rate = tf.placeholder(tf.float32)
    train_step1 = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(error1, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope))
    train_step2 = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(error2, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope))
    # train_step1 = tf.train.AdamOptimizer(1e-4).minimize(error1)
    # train_step2 = tf.train.AdamOptimizer(1e-4).minimize(error2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rate = 1e-2
        for i in range(20000):
        # for i in range(100):
            gene_batch = fake_samples(batch_size)
            real_batch = samples(batch_size)
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
                print('step %d, d_loss: %g, g_loss: %g'
                      % (i, train_error1, train_error2))
            if i % 100 == 0:
                generated = sess.run([tf.reshape(go, [-1])], feed_dict={
                    ge: fake_samples(batch_size)
                })
                plt.hist(generated, normed=True)
                generated = np.asarray(generated)
                upper = np.max(generated)
                lower = np.min(generated)
                upper = np.max([abs(lower), abs(upper)])
                lower = - upper
                x = np.linspace(lower, upper, 1000)
                miu, sigma = 0., 1.0
                y1 = 1. / math.sqrt(2 * math.pi) / sigma / np.exp(np.square(x - miu) / 2. / sigma / sigma)
                plt.plot(x, y1, 'r-')
                plt.savefig("s.png")
                plt.close()
        # plt.show()
        # if i % 10 == 0:
                # for i in range(10):
                #     image = real_batch[i] * 256
                #     png = sess.run(tf.image.encode_png(np.reshape(image, [28, 28, 1])))
                #     with open("%dr.png" % i, "wb") as f:
                #         f.write(png)
                # generated = fake_sample(10)
                # decoded = sess.run(tf.cast(generator.generate(generated) * 256, tf.uint8))
                # _re, decoded = sess.run([re, ro * 256], feed_dict={rx: real_batch})
                # for i in range(10):
                #     image = decoded[i]
                #     png = sess.run(tf.image.encode_png(tf.cast(tf.reshape(image, [28, 28, 1]), tf.uint8)))
                #     with open("%d.png" % i, "wb") as f:
                #         f.write(png)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                                            default='./mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
