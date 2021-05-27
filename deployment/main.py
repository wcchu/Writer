from flask import Flask
from learn import build_model, CHECKPOINT_DIR, EMBEDDING_SIZE, RNN_UNITS
# TODO: move constants to env variables
import pickle
import tensorflow as tf
app = Flask(__name__)

# prediction
SEED_TEXT = "In the beginning"
MIN_LEN = 200
MAX_LEN = 1000
TEMPERATURE = 1.0


def build_prediction_model(nc):
    # Rebulid model with batch size = 1
    model = build_model(n_chars=nc,
                        emb_size=EMBEDDING_SIZE,
                        rnn_units=RNN_UNITS,
                        batch_size=1)
    # import trained weights
    model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    # model for prediction
    model.build(tf.TensorShape([1, None]))
    return model


def writer(model, seed, lmin, lmax, temp, char_to_id, id_to_char):
    '''Write new text'''
    # convert seed text to id list
    input_ids = tf.expand_dims([char_to_id[c] for c in seed], 0)

    # initial written text
    written = seed

    model.reset_states()
    for k in range(lmax):
        pred = tf.squeeze(model(input_ids), 0) / temp

        # predict the last id returned by the model
        pred_id = tf.random.categorical(pred, num_samples=1)[-1, 0].numpy()

        # pass predicted ids as the input of the next prediction
        input_ids = tf.expand_dims([pred_id], 0)

        # get the last character and attach it to text
        pred_char = id_to_char[pred_id]
        written = written + pred_char

        # if at least lmin and on a period (.), stop writing
        if len(written) >= lmin and pred_char == ".":
            break

    return written


@app.route('/')
@app.route('/<float:temp>')
@app.route('/<float:temp>/<string:seed_text>')
@app.route('/<float:temp>/<string:seed_text>/<int:min_len>')
@app.route('/<float:temp>/<string:seed_text>/<int:min_len>/<int:max_len>')
def write(temp=None, seed_text=None, min_len=None, max_len=None):

    seed_text = seed_text or SEED_TEXT
    min_len = min_len or MIN_LEN
    max_len = max_len or MAX_LEN
    temp = temp or TEMPERATURE

    # load pickles
    n_chars = pickle.load(open('n_chars.pk', 'rb'))
    char_to_id = pickle.load(open('char_to_id.pk', 'rb'))
    id_to_char = pickle.load(open('id_to_char.pk', 'rb'))

    # generate new text
    model = build_prediction_model(n_chars)

    # # Execute writing
    new_text = writer(model, seed_text, min_len, max_len, temp, char_to_id,
                      id_to_char)
    return new_text


if __name__ == '__main__':
    app.run()
