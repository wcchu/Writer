from learn import build_model, CHECKPOINT_DIR, EMBEDDING_SIZE, RNN_UNITS
import tensorflow as tf
import pickle
from flask import Flask
app = Flask(__name__)


# prediction
SEED_TEXT = "To be honest,"
WRITTEN_LEN = 200
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


def writer(model, seed, length, temp, char_to_id, id_to_char):
    '''Write new text'''
    # convert seed text to id list
    input_ids = tf.expand_dims([char_to_id[c] for c in seed], 0)

    # storage for written text
    written = []

    model.reset_states()
    for k in range(length):
        pred = tf.squeeze(model(input_ids), 0) / temp

        # predict the last id returned by the model
        pred_id = tf.random.categorical(pred, num_samples=1)[-1, 0].numpy()

        # pass predicted ids as the input of the next prediction
        input_ids = tf.expand_dims([pred_id], 0)

        written.append(id_to_char[pred_id])

    return (seed + "".join(written))


@app.route('/')
@app.route('/<string:seed_text>')
@app.route('/<string:seed_text>/<int:written_len>')
@app.route('/<string:seed_text>/<int:written_len>/<float:temperature>')
def write(seed_text=None, written_len=None, temperature=None):

    seed_text = seed_text or SEED_TEXT
    written_len = written_len or WRITTEN_LEN
    temperature = temperature or TEMPERATURE

    # load pickles
    n_chars = pickle.load(open('n_chars.pk', 'rb'))
    char_to_id = pickle.load(open('char_to_id.pk', 'rb'))
    id_to_char = pickle.load(open('id_to_char.pk', 'rb'))

    # generate new text
    model = build_prediction_model(n_chars)

    # # Execute writing
    new_text = writer(model, seed_text, written_len, temperature, char_to_id,
                      id_to_char)
    return new_text


if __name__ == '__main__':
    app.run()
