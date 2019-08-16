import tensorflow as tf
import numpy as np
import collections
import os

### Parameters

# model
CHECKPOINT_DIR = './writer_checkpoints'
EMBEDDING_SIZE = 256
RNN_UNITS = 1024

# training
DATA_DIR = "bible.txt"
EPOCHS = 1
TIME_STEPS = 200
BATCH_SIZE = 64
BUFFER_SIZE = 10000

# prediction
SEED_TEXT = "To be honest,"
WRITTEN_LEN = 200
TEMPERATURE = 1.0

# Import data
def get_data(data_path=None):
    """Load raw data from data directory "data_path".
    Reads text file, converts strings to integer ids
    Args:
    data_path: string path to the directory
    Returns:
    tuple (raw_data, vocabulary)
    """

    data = list(open(data_path, "r").read())
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    chars, _ = list(zip(*count_pairs))
    char_to_id = {c:i for i, c in enumerate(chars)}
    id_to_char = {i:c for i, c in enumerate(chars)}

    data_in_ids = [char_to_id[char] for char in data]
    return data, data_in_ids, char_to_id, id_to_char


raw_data_chars, raw_data_ids, char_to_id, id_to_char = get_data(DATA_DIR)
n_chars = len(char_to_id)
print(n_chars)


# Prepare training data
### Create examples

# create tf.Dataset object
raw_dataset = tf.data.Dataset.from_tensor_slices(raw_data_ids)

# create examples
# NOTE: the "batch" defined here is one example (batch of characters) instead of batch of examples
examples = raw_dataset.batch(TIME_STEPS+1, drop_remainder=True)


### Map (to inputs and responses), shuffle, and batch data
def input_and_response(example):
    example_input = example[:-1]
    example_response = example[1:]
    return example_input, example_response

dataset = examples.map(input_and_response).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build model

### Set up model
def build_model(n_chars, emb_size, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            n_chars,
            emb_size,
            batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            stateful=True,
            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(n_chars)
    ])
    return model

model = build_model(
    n_chars=n_chars,
    emb_size=EMBEDDING_SIZE,
    rnn_units=RNN_UNITS,
    batch_size=BATCH_SIZE)

### Define loss and optimizer
def loss(responses, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(responses, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Train model

### Configure checkpoints
# Name of the checkpoint files
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

### Execute training
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

model.summary()

# Generate new text
### Rebulid model with batch size = 1
model = build_model(
    n_chars=n_chars,
    emb_size=EMBEDDING_SIZE,
    rnn_units=RNN_UNITS,
    batch_size=1)

model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))

model.build(tf.TensorShape([1, None]))


### Define writer
def writer(model, seed, length, temp):

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

### Execute writing
writer(model, SEED_TEXT, WRITTEN_LEN, TEMPERATURE)
