import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

real_miu = 0.8
real_sigma = 15

samples = np.random.normal(real_miu, real_sigma, 1000).reshape((-1, 1))

_miu = tf.Variable(tf.random_normal([1], 0.0, 10.0, dtype=tf.float64), dtype=tf.float64)
_sigma = tf.exp(tf.Variable(tf.random_normal([1], 0.0, 1.0, dtype=tf.float64), dtype=tf.float64))

batch_size = 1
_batch = tf.placeholder(tf.float64, (batch_size, 1))

# _minus_log_likelihood1 = tf.reduce_sum(tf.log(_sigma) + tf.square(_batch - _miu) / 2.0 / _sigma / _sigma)
_learning_rate = tf.placeholder(tf.float64)
_minus_log_likelihood2 = - tf.reduce_sum(tf.log(1.0 / _sigma / tf.exp(tf.square(_batch - _miu) / 2.0 / _sigma / _sigma)))
_optimizer = tf.train.GradientDescentOptimizer(_learning_rate).minimize(_minus_log_likelihood2)

def sample(samples, n):
    return samples[np.random.randint(0, len(samples), n)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    miu, sigma = 100, math.exp(10)
    # for i in range(1000):
    i = 1
    # while abs(miu - real_miu) > 0.01 or abs(sigma - real_sigma) > 0.01:
    lr = 0.01
    while i < 10**6:
        _, likely, miu, sigma = sess.run([_optimizer, _minus_log_likelihood2, _miu, _sigma], feed_dict={_batch: sample(samples, batch_size), _learning_rate: lr})
        # sampled = sample(samples, batch_size)
        # likely1, likely2, miu, sigma = sess.run([_minus_log_likelihood1, _minus_log_likelihood2, _miu, _sigma], feed_dict={_batch: sampled})
        if i % 100 == 0:
            # print i, likely1, likely2, miu, sigma, sampled
            print i, likely, miu, sigma
        if i % 1000 == 0:
            lr /= 1.005
        i += 1
    # print i, likely, miu, sigma

plt.hist(samples, normed=True)

upper = max(samples)
lower = min(samples)
upper = max(abs(lower), abs(upper))
lower = - upper
x = np.linspace(lower, upper, 1000)
y0 = 1. / math.sqrt(2 * math.pi) / real_sigma / np.exp(np.square(x - real_miu) / 2. / real_sigma / real_sigma)
y1 = 1. / math.sqrt(2 * math.pi) / sigma / np.exp(np.square(x - miu) / 2. / sigma / sigma)
plt.plot(x, y0, 'g-')
plt.plot(x, y1, 'r-')

plt.show()
