import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

real_alpha = [0.4, 0.6]
real_miu = [-2.0, 1.0]
real_sigma = [1., 2.]

count = 1000
choices = np.random.choice(len(real_alpha), count, p=real_alpha)
samples = np.asarray([], dtype=np.float64)
for i in range(len(real_alpha)):
    samples = np.concatenate([samples, np.random.normal(real_miu[i], real_sigma[i], (choices == i).sum())])

K = 10
_alpha = tf.exp(tf.Variable(tf.zeros([K], dtype=tf.float64), dtype=tf.float64))
_alpha = _alpha / tf.reduce_sum(_alpha)
_miu = tf.Variable(tf.random_normal([K], 0.0, 1.0, dtype=tf.float64), dtype=tf.float64)
_sigma = tf.exp(tf.Variable(tf.random_normal([K], -1.0, 1.0, dtype=tf.float64), dtype=tf.float64))

batch_size = 16
_batch = tf.placeholder(tf.float64, [batch_size])

_neg_log_likelihood = 0
for i in range(batch_size):
    _neg_log_likelihood -= tf.log(tf.reduce_sum(_alpha / _sigma / tf.exp(tf.square(_batch[i] - _miu) / 2.0 / _sigma / _sigma)))
_optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(_neg_log_likelihood)

def sample(samples, n):
    return samples[np.random.randint(0, len(samples), n)]

while True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        IC = 10**5
        for i in xrange(IC):
            # likely, alpha, miu, sigma = sess.run([_neg_log_likelihood, _alpha, _miu, _sigma], feed_dict={_batch: sample(samples, batch_size)})
            # print i, likely, alpha, miu, sigma
            _, likely, alpha, miu, sigma = sess.run([_optimizer, _neg_log_likelihood, _alpha, _miu, _sigma], feed_dict={_batch: sample(samples, batch_size)})
            print i, likely, alpha, miu, sigma
            if math.isnan(likely):
                break
        if i+1 == IC:
            break

# print samples
plt.hist(samples, normed=True)
# plt.show()

upper = max(samples)
lower = min(samples)
upper = max(abs(lower), abs(upper))
lower = - upper
x = np.linspace(lower, upper, count)
# y0 = 1. / math.sqrt(2 * math.pi) / real_sigma / np.exp(np.square(x - real_miu) / 2. / real_sigma / real_sigma)
y1 = np.zeros(count, dtype=np.float64);
for i in range(len(real_alpha)):
    y1 += real_alpha[i] / math.sqrt(2 * math.pi) / real_sigma[i] / np.exp(np.square(x - real_miu[i]) / 2. / real_sigma[i] / real_sigma[i])
y2 = np.zeros(count, dtype=np.float64)
for i in range(len(alpha)):
    y2 += alpha[i] / math.sqrt(2 * math.pi) / sigma[i] / np.exp(np.square(x - miu[i]) / 2. / sigma[i] / sigma[i])
# plt.plot(x, y0, 'g-')
plt.plot(x, y1, 'ro')
plt.plot(x, y2, 'bo')
plt.show()