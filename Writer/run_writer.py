"""
Run a trained writer
"""

import logging
from optparse import OptionParser
import tensorflow as tf
import numpy as np
import json


def get_seed(data_path=None):
    """Load seed text from data directory "data_path"."""
    words = tf.gfile.GFile(
        data_path, "r").read().replace("\n", "<eos> ").replace(".", "").replace(
            ",", "").replace(";", "").split()
    return words


def run():
    """run"""

    # parse option(s)
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input', default='bible_seed.txt')
    (options, args) = parser.parse_args()

    # load word_to_id and id_to_word from file
    with open('word_to_id.txt', 'r') as f:
        word_to_id = json.loads(f.read())
    with open('id_to_word.txt', 'r') as f:
        id_to_word = json.loads(f.read())
    id_to_word = {int(k): v for (k, v) in id_to_word.items()}

    n_words = len(word_to_id)

    time_steps = 10

    seed_words = get_seed(options.input)

    seed_ids = list()
    i = 0
    while len(seed_ids) < 10:
        if seed_words[i] in word_to_id:
            seed_ids.append(word_to_id[seed_words[i]])
        i += 1
    seed_ids = np.array(seed_ids)
    print(seed_ids)
    print([id_to_word[id] for id in seed_ids])

    initial_seed = np.eye(n_words)[seed_ids].reshape(1, time_steps, n_words)

    X = tf.placeholder(tf.float32, [None, time_steps, n_words])

    cell = tf.nn.rnn_cell.GRUCell(n_words)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    saver = tf.train.Saver()

    pred_iterations = 20
    seed = initial_seed.copy()
    with tf.Session() as sess:

        saver.restore(sess, "model/")

        for iteration in range(pred_iterations):
            x_pred = seed[:, -time_steps:, :]
            x_ind = np.argmax(x_pred, axis=2).reshape(time_steps)
            print(x_ind)
            pred_probs = sess.run(outputs, feed_dict={X: x_pred})
            pred_last_index = np.argmax(pred_probs[:, -1:, :], axis=2)
            pred_last_word = id_to_word[pred_last_index[0][0]]
            print("iteration = %5d, word = %s" % (iteration, pred_last_word))
            seed_to_add = np.eye(n_words)[pred_last_index]
            seed = np.append(seed, seed_to_add, axis=1)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
