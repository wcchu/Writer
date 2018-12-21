"""
Run a trained writer
"""

import logging
import tensorflow as tf
import numpy as np
import json


def run():
    """run"""

    # load word_to_id and id_to_word from file
    with open('word_to_id.txt', 'r') as f:
        word_to_id = json.loads(f.read())
    with open('id_to_word.txt', 'r') as f:
        id_to_word = json.loads(f.read())
    id_to_word = {int(k): v for (k, v) in id_to_word.items()}

    n_words = len(word_to_id)

    time_steps = 10

    seed_ids = np.random.randint(0, n_words, 10)

    print(seed_ids)

    seed_sentence = [id_to_word[id] for id in seed_ids]
    print(seed_sentence)
    initial_seed = np.eye(n_words)[seed_ids].reshape(1, time_steps, n_words)

    X = tf.placeholder(tf.float32, [None, time_steps, n_words])

    cell = tf.nn.rnn_cell.GRUCell(n_words)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    saver = tf.train.Saver()

    pred_iterations = 10
    seed = initial_seed.copy()
    with tf.Session() as sess:

        saver.restore(sess, "./model/")

        for iteration in range(pred_iterations):
            x_pred = seed[:, -time_steps:, :]
            pred_probs = sess.run(outputs, feed_dict={X: x_pred})
            pred_last_index = np.argmax(pred_probs[:, -1:, :], axis=2)
            pred_last_word = id_to_word[pred_last_index[0][0]]
            print("iteration = %5d, word = %s" % (iteration, pred_last_word))
            seed_to_add = np.eye(n_words)[pred_last_index]
            seed = np.append(seed, seed_to_add, axis=1)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
