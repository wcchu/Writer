"""
Train a writer
"""

import logging
from optparse import OptionParser
import tensorflow as tf
import numpy as np
import collections
import json


def get_data(data_path=None):
    """Load raw data from data directory "data_path".
    Reads text file, converts strings to integer ids
    Args:
    data_path: string path to the directory
    Returns:
    tuple (raw_data, vocabulary)
    """

    data = tf.gfile.GFile(
        data_path, "r").read().replace("\n", "<eos> ").replace(".", "").replace(
            ",", "").replace(";", "").split()

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))

    data_in_ids = [word_to_id[word] for word in data if word in word_to_id]
    return data, data_in_ids, word_to_id, id_to_word


def vec_to_word(vec, id_to_word):
    """
    Convert a one-hot array into a word
    """
    index = np.argmax(vec, axis=0)  # get the index of the most probable word
    word = id_to_word[index]
    return word


# build a random batch from data
def get_batch(data, batch_size, time_steps, input_size):
    """
    Build a random batch from data
    """
    batch = np.zeros([batch_size, time_steps + 1, input_size])
    for row in range(batch_size):
        t0 = np.random.randint(0, len(data) - time_steps)  # starting time
        batch[row, :, :] = np.eye(input_size)[data[t0:t0 + time_steps + 1]]
    return batch[:, :-1, :], batch[:, 1:, :]


def run():
    """run"""

    # parse option(s)
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input', default='bible.txt')
    (options, args) = parser.parse_args()

    # import data from disk
    raw_data_words, raw_data_ids, word_to_id, id_to_word = get_data(
        data_path=options.input)
    n_words = len(word_to_id)
    print("number of distinct words = %d" % n_words)
    # write word_to_id and id_to_word to files
    with open('word_to_id.txt', 'w') as f:
        f.write(json.dumps(word_to_id))
    with open('id_to_word.txt', 'w') as f:
        f.write(json.dumps(id_to_word))

    # set training parameters
    batch_size = 10
    time_steps = 10
    # epochs = 1 # not considering epoch now
    iterations = 100
    learning_rate = 0.1

    # Input / Output(target)
    X = tf.placeholder(tf.float32, [None, time_steps, n_words])
    Y = tf.placeholder(tf.float32, [None, time_steps, n_words])

    # Setup RNN
    cell = tf.nn.rnn_cell.GRUCell(n_words)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    # option1: "outputs"
    # loss = tf.reduce_mean(tf.square(outputs - Y))

    # option2: "probs"
    dense = tf.layers.dense(inputs=outputs, units=n_words, activation=None)
    probs = tf.nn.softmax(dense)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=probs))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Run training
    with tf.Session() as sess:
        sess.run(init)

        for iteration in range(iterations):

            x_batch, y_batch = get_batch(
                data=raw_data_ids,
                batch_size=batch_size,
                time_steps=time_steps,
                input_size=n_words)

            sess.run(train, feed_dict={X: x_batch, Y: y_batch})

            if iteration % 10 == 0:
                loss_ = loss.eval(feed_dict={X: x_batch, Y: y_batch})
                print("iteration = %5d, loss = %10.4f" % (iteration, loss_))

        # Save model
        saver.save(sess, "./model/")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
