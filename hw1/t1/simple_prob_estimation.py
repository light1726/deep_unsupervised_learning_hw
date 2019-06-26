import numpy as np
import tensorflow as tf


def sample_data():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    return np.digitize(samples, np.linspace(0.0, 1.0, 100))


class DensityEstimator(object):
    def __init__(self, dim=100, name='density_estimatation'):
        self.dim = dim
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.theta = tf.get_variable(name='theta',
                                         shape=[self.dim, 1],
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer)

    def __call__(self, inputs):
        # inputs: [batch,] range from 1 - 100
        logits = tf.gather(self.theta, inputs)
        probs = tf.exp(logits) / tf.reduce_sum(tf.exp(self.theta))
        return tf.reduce_mean(-tf.log(probs)/tf.log(2.0))


def train():
    data = sample_data()
    data_len = data.size
    train_data = data[:int(0.8*data_len)]
    dev_data = data[int(0.7*data_len): int(0.8*data_len)]
    test_data = data[int(0.8*data_len):]
    batch_size = 16
    # set up graph
    input_pl = tf.placeholder(dtype=tf.int32, shape=[None, ], name='input_pl')
