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
    data = list(tf.gfile.GFile(data_path, "r").read().replace("\n", ""))
    return data


def run():
    """run"""

    # parse option(s)
    parser = OptionParser()
    parser.add_option('-i', '--input', dest='input', default='bible_seed.txt')
    (options, args) = parser.parse_args()

    # load char_to_id and id_to_char from file
    with open('char_to_id.txt', 'r') as f:
        char_to_id = json.loads(f.read())
    with open('id_to_char.txt', 'r') as f:
        id_to_char = json.loads(f.read())
    id_to_char = {int(k): v for (k, v) in id_to_char.items()}

    n_chars = len(char_to_id)

    time_steps = 100

    seed_chars = get_seed(options.input)
    # limit the length of seed to time_steps
    seed_chars = seed_chars[:time_steps]
    seed_ids = [char_to_id[char] for char in seed_chars]
    # print seed as a sentence
    print("".join(seed_chars))

    X = tf.placeholder(tf.float32, [None, time_steps, n_chars])

    cell = tf.nn.rnn_cell.GRUCell(n_chars)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    saver = tf.train.Saver()

    pred_iterations = 100
    new_seed_ids = seed_ids.copy()
    with tf.Session() as sess:

        saver.restore(sess, "./model/")

        for iteration in range(pred_iterations):
            x_pred_ids = new_seed_ids[-time_steps:]
            x_pred = np.eye(n_chars)[x_pred_ids].reshape(1, time_steps, n_chars)
            pred_probs = sess.run(outputs, feed_dict={X: x_pred})
            pred_last_index = np.argmax(pred_probs[:, -1:, :], axis=2)[0][0]
            pred_last_char = id_to_char[pred_last_index]
            new_seed_ids = np.append(new_seed_ids, pred_last_index)

    final_sentence = "".join([id_to_char[id] for id in new_seed_ids])
    print(final_sentence)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
