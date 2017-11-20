from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

batch_size = 16*4*4*2

d_scope = "d"
g_scope = "g"


class NaiveDiscriminator:
    def __init__(self):
        with tf.variable_scope(d_scope):
            self.W_fc1 = weight_variable([28*28, 100])
            self.b_fc1 = bias_variable([100])
            self.W_fc2 = weight_variable([100, 2])
            self.b_fc2 = bias_variable([2])

    def discriminate(self, x):
        x = tf.reshape(x, [-1, 28*28])

        x = tf.matmul(x, self.W_fc1) + self.b_fc1
        x = tf.nn.tanh(x) * 3

        x = tf.matmul(x, self.W_fc2) + self.b_fc2
        x = tf.nn.sigmoid(x)

        return x


class Discriminator:
    def __init__(self):
        with tf.variable_scope(d_scope):
            self.W_conv1 = weight_variable([6, 6, 1, 32])
            self.b_conv1 = bias_variable([32])

            self.W_conv2 = weight_variable([5, 5, 32, 64])
            self.b_conv2 = bias_variable([64])

            self.W_conv3 = weight_variable([4, 4, 64, 128])
            self.b_conv3 = bias_variable([128])

            self.W_conv4 = weight_variable([3, 3, 128, 256])
            self.b_conv4 = bias_variable([256])

            self.W_conv5 = weight_variable([2, 2, 256, 512])
            self.b_conv5 = bias_variable([512])

            self.W_fc1 = weight_variable([512, 1024])
            self.b_fc1 = bias_variable([1024])

            self.W_fc2 = weight_variable([1024, 2])
            self.b_fc2 = bias_variable([2])

    def discriminate(self, x):
        x = tf.reshape(x, [-1, 28, 28, 1])

        x = tf.nn.conv2d(x, self.W_conv1, [1, 2, 2, 1], "SAME") + self.b_conv1
        x = tf.nn.elu(x)

        x = tf.nn.conv2d(x, self.W_conv2, [1, 2, 2, 1], "SAME") + self.b_conv2
        x = tf.nn.elu(x)

        x = tf.nn.conv2d(x, self.W_conv3, [1, 2, 2, 1], "SAME") + self.b_conv3
        x = tf.nn.elu(x)

        x = tf.nn.conv2d(x, self.W_conv4, [1, 2, 2, 1], "SAME") + self.b_conv4
        x = tf.nn.elu(x)

        x = tf.nn.conv2d(x, self.W_conv5, [1, 2, 2, 1], "SAME") + self.b_conv5
        x = tf.nn.elu(x)

        x = tf.reshape(x, [int(x.get_shape()[0]), -1])

        x = tf.matmul(x, self.W_fc1) + self.b_fc1
        x = tf.nn.tanh(x)

        x = tf.matmul(x, self.W_fc2) + self.b_fc2

        return x


class NaiveGenerator:
    def __init__(self):
        with tf.variable_scope(g_scope):
            self.W_fc1 = weight_variable([100, 256])
            self.b_fc1 = bias_variable([256])
            self.W_fc2 = weight_variable([256, 256])
            self.b_fc2 = bias_variable([256])
            self.W_fc3 = weight_variable([256, 28*28])
            self.b_fc3 = bias_variable([28*28])

    def generate(self, x):
        x = tf.matmul(x, self.W_fc1) + self.b_fc1
        x = tf.nn.tanh(x) * 3

        x = tf.matmul(x, self.W_fc2) + self.b_fc2
        x = tf.nn.tanh(x) * 3

        x = tf.matmul(x, self.W_fc3) + self.b_fc3
        x = tf.nn.sigmoid(x)

        return x


class DeconvolutionalGenerator:
    def __init__(self):
        self.repeat = 3
        with tf.variable_scope(g_scope):
            self.W_fc1 = [weight_variable([100, 100]) for i in range(self.repeat)]
            self.b_fc1 = [bias_variable([100]) for i in range(self.repeat)]
            self.s_fc1 = [scale_variable([100]) for  i in range(self.repeat)]
            self.m_fc1 = [mean_variable([100]) for  i in range(self.repeat)]

            self.W_fc2 = weight_variable([100, 7*7])
            self.b_fc2 = bias_variable([7*7])
            self.s_fc2 = scale_variable([7*7])
            self.m_fc2 = mean_variable([7*7])

            self.W_dconv1 = weight_variable([7, 7, 50, 100])
            self.b_dconv1 = bias_variable([50])
            self.s_dconv1 = scale_variable([50])
            self.m_dconv1 = mean_variable([50])

            self.W_dconv2 = weight_variable([6, 6, 25, 50])
            self.b_dconv2 = bias_variable([25])
            self.s_dconv2 = scale_variable([25])
            self.m_dconv2 = mean_variable([25])

            self.W_dconv3 = weight_variable([5, 5, 10, 25])
            self.b_dconv3 = bias_variable([10])
            self.s_dconv3 = scale_variable([10])
            self.m_dconv3 = mean_variable([10])

            self.W_dconv4 = weight_variable([4, 4, 1, 10])
            self.b_dconv4 = bias_variable([1])
            self.s_dconv4 = scale_variable([1])
            self.m_dconv4 = mean_variable([1])

            self.W_fc3 = weight_variable([100, 256])
            self.b_fc3 = bias_variable([256])

            # self.activator = tf.nn.tanh
            # self.activator = lambda _: _
            # self.activator = tf.sin
            # self.activator = lambda _: 4 * (tf.nn.sigmoid(_) - 0.5)
            self.activator = lambda _: 2 * tf.nn.tanh(_)

    def map(self, x):
        for i in range(self.repeat):
            x = tf.matmul(x, self.W_fc1[i]) + self.b_fc1[i]
            x = (self.activator(x) - self.m_fc1[i]) * self.s_fc1[i]
        return x

    def generate(self, x):
        x = self.map(x)
        x = self.restore(x)
        return x

    def restore(self, x):
        with tf.variable_scope(g_scope):
            x = tf.reshape(x, [int(x.get_shape()[0]), 1, 1, -1])

            x = tf.nn.conv2d_transpose(x, self.W_dconv1, [int(x.get_shape()[0]), 7, 7, 50], [1, 7, 7, 1], "SAME") + self.b_dconv1
            x = (self.activator(x) - self.m_dconv1) * self.s_dconv1

            x = tf.nn.conv2d_transpose(x, self.W_dconv2, [int(x.get_shape()[0]), 14, 14, 25], [1, 2, 2, 1], "SAME") + self.b_dconv2
            x = (self.activator(x) - self.m_dconv2) * self.s_dconv2

            x = tf.nn.conv2d_transpose(x, self.W_dconv3, [int(x.get_shape()[0]), 28, 28, 10], [1, 2, 2, 1], "SAME") + self.b_dconv3
            x = (self.activator(x) - self.m_dconv3) * self.s_dconv3

            x = tf.nn.conv2d_transpose(x, self.W_dconv4, [int(x.get_shape()[0]), 28, 28, 1], [1, 1, 1, 1], "SAME") + self.b_dconv4

            x = tf.nn.sigmoid(x)
            x /= tf.reduce_max(x)

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


def scale_variable(shape):
    """scale_variable generates a bias variable of a given shape."""
    initial = tf.random_uniform(shape, 1., 2.)
    return tf.Variable(initial)


def mean_variable(shape):
    """mean_variable generates a bias variable of a given shape."""
    initial = tf.random_uniform(shape, 0.0, 0.1)
    return tf.Variable(initial)


def fake_embedding(n):
    return np.asarray(np.random.uniform(-1, 1, (n, 100)), np.float32)


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    discriminator = Discriminator()
    generator = DeconvolutionalGenerator()

    rx = tf.placeholder(tf.float32, [batch_size, 784])
    ry = discriminator.discriminate(rx)

    ge = tf.placeholder(tf.float32, [batch_size, 100])
    go = generator.generate(ge)
    gy = discriminator.discriminate(go)

    tl = np.ones([batch_size, 2], np.float32) * [0.9, 0]
    fl = np.ones([batch_size, 2], np.float32) * [0, 1.0]

    loss1 = \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tl, logits=ry)) + \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=fl, logits=gy))

    loss2 = \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tl, logits=gy))

    learning_rate = tf.placeholder(tf.float32)
    train_step1 = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(loss1, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope))
    train_step2 = tf.train.GradientDescentOptimizer(learning_rate).\
        minimize(loss2, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope))

    with tf.Session() as sess:
        rd = 1e-2*1.02
        rg = 1e-2
        train_loss1 = train_loss2 = 1.
        sess.run(tf.global_variables_initializer())
        steps = 2000
        for i in range(steps):
            batch = mnist.train.next_batch(batch_size, shuffle=True)
            gene_batch = fake_embedding(batch_size)
            real_batch = batch[0]
            if i % 10 == 0:
                rd /= 1.001
                rg /= 1.001
            train_loss1, _ = \
                sess.run([loss1, train_step1],
                                 feed_dict={
                                     learning_rate: rd,
                                     rx: real_batch, ge: gene_batch,
                                 })
            train_loss2, _ = \
                sess.run([loss2, train_step2],
                                 feed_dict={
                                     learning_rate: rg,
                                     rx: real_batch, ge: gene_batch,
                                 })
            print('step %d, d_loss: %g, g_loss: %g'
                  % (i, train_loss1, train_loss2))

            if i % 10 == 0:
                decoded = sess.run(tf.cast(go * 255, tf.uint8), feed_dict={
                    ge: gene_batch
                })
                for j in range(16):
                    image = decoded[j]
                    png = sess.run(tf.image.encode_png(tf.cast(tf.reshape(image, [28, 28, 1]), tf.uint8)))
                    with open("img/%d.png" % j, "wb") as f:
                        f.write(png)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                                            default='./mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
