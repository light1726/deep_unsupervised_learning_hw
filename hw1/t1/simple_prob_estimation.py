import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


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
        return tf.reduce_mean(-tf.log(probs) / tf.log(2.0))

    def sample(self, n_sample):
        distribution = tf.distributions.Categorical(logits=tf.squeeze(self.theta))
        samples = distribution.sample([n_sample])
        return samples

    def distributions(self):
        return tf.nn.softmax(tf.squeeze(self.theta))


def main():
    data = sample_data()
    data_len = data.size
    train_data = data[:int(0.8 * data_len)]
    train_len = train_data.size
    dev_data = data[int(0.7 * data_len): int(0.8 * data_len)]
    test_data = data[int(0.8 * data_len):]
    batch_size = 16
    train_epochs = 100
    learning_rate = 1e-3
    eval_duration = 20
    train_logdir = '.\\log\\train'
    dev_logdir = '.\\log\\dev'

    # set up graph
    input_pl = tf.placeholder(dtype=tf.int32, shape=[None, ], name='input_pl')
    n_sample_pl = tf.placeholder(dtype=tf.int32, shape=[], name='n_sample')
    density_estimator = DensityEstimator()
    draw_samples = density_estimator.sample(n_sample_pl)
    distributions = density_estimator.distributions()
    mean_minus_log_probs = density_estimator(input_pl)
    tf.summary.scalar('mean_minus_log_probs', mean_minus_log_probs)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(mean_minus_log_probs)
    train_writer = tf.summary.FileWriter(train_logdir)
    train_writer.add_graph(tf.get_default_graph())
    dev_writer = tf.summary.FileWriter(dev_logdir)
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver()

    # train
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_steps_per_eopch = train_len // batch_size - 1
    global_step = 0
    for epoch in range(train_epochs):
        for i in range(train_steps_per_eopch):
            global_step = epoch * train_steps_per_eopch + i
            batch_data = train_data[i * batch_size:(i + 1) * batch_size]
            loss, _, summary = sess.run([mean_minus_log_probs, train_op, summaries],
                                        feed_dict={input_pl: batch_data})
            train_writer.add_summary(summary, global_step=global_step)
            print("step: {}, train_loss: {}".format(global_step, loss))
            if global_step % eval_duration == 0:
                eval_loss, summary = sess.run([mean_minus_log_probs, summaries],
                                              feed_dict={input_pl: dev_data})
                print("step: {}, eval_loss: {}".format(global_step, eval_loss))
                dev_writer.add_summary(summary, global_step=global_step)
                saver.save(sess, save_path=os.path.join(train_logdir, 'model.ckpt'),
                           global_step=global_step)
                print('Saved checkpoint to {}'.format(train_logdir))
    saver.save(sess, save_path=os.path.join(train_logdir, 'model.ckpt'),
               global_step=global_step)
    print('Saved final checkpoint to {}'.format(train_logdir))
    test_loss = sess.run(mean_minus_log_probs, feed_dict={input_pl: test_data})
    print("Test loss is {}".format(test_loss))
    samples = sess.run(draw_samples, feed_dict={n_sample_pl: 1000})
    probs = sess.run(distributions)
    sess.close()

    plt.subplot(311)
    plt.hist(data, bins=100)
    plt.subplot(312)
    plt.hist(samples, bins=100)
    plt.subplot(313)
    plt.bar(np.arange(100), probs)
    plt.show()


if __name__ == '__main__':
    main()
