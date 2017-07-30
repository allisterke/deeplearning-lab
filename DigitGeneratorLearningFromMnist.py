import numpy as np
import tensorflow as tf
import math
import cv2

# trd = np.fromfile('/home/ubuntu/datasets/mnist/train-images-idx3-ubyte', dtype=np.uint8)[16:]
# trd = trd.reshape((-1, 28 * 28)) / 255.0

def read_samples():
    ted = np.fromfile('/home/ubuntu/datasets/mnist/t10k-images-idx3-ubyte', dtype=np.uint8)[16:]
    ted = ted.reshape((-1, 28 * 28)) / 255.0
    tel = np.fromfile('/home/ubuntu/datasets/mnist/t10k-labels-idx1-ubyte', dtype=np.uint8)[8:]
    return ted[tel == 6, :]

samples = read_samples()

D = 28*28

def sample(t, n):
    return t[np.random.randint(0, t.shape[0], n), :D]

K = 10*2

_alpha = tf.exp(tf.Variable(np.zeros(K), dtype=tf.float64))
_alpha = _alpha / tf.reduce_sum(_alpha)
_sigma = tf.exp(tf.Variable(np.zeros((K, D)), dtype=tf.float64))
_miu = tf.Variable(np.ones((K, D)) * 0.5, dtype=tf.float64)

batch_size = 16
_batch = tf.placeholder(tf.float64, [batch_size, D])

_neg_log_likelihood = 0
for i in range(batch_size):
    _neg_log_likelihood -= tf.reduce_sum(tf.log(tf.reduce_sum(tf.stack([_alpha] * D, 1) / _sigma / tf.exp(tf.square(tf.stack([_batch[0, :]] * K) - _miu) / 2. / _sigma / _sigma), [0])))
_optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(_neg_log_likelihood)

while True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        IC = 10**4*6
        for i in range(IC):
            # _, likelihood, alpha, sigma, miu = sess.run([_optimizer, _neg_log_likelihood, _alpha, _sigma, _miu], feed_dict={_batch: sample(trd, batch_size)})
            # print i, likelihood , alpha, sigma, miu
            _, likelihood = sess.run([_optimizer, _neg_log_likelihood], feed_dict={_batch: sample(samples, batch_size)})
            print(i, likelihood)
            if math.isnan(likelihood):
                break
        # if i+1 == IC:
        if True:
            alpha, sigma, miu = sess.run([_alpha, _sigma, _miu])
            break

for i in range(20):
    index = np.random.choice(len(alpha), 1, p=alpha)
    image = np.asarray([np.random.normal(miu[index, j], sigma[index, j]) for j in range(28*28)]).reshape((28, 28)) * 255
    cv2.imwrite('%d.png' % (i,), image)
