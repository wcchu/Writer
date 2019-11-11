# -*- coding: utf-8 -*-
"""
Build PairShift model with TensorFlow 2.0
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_FILE = 'input.csv'
EPOCHS = 5
CEN_VALUE = 25.0


def import_data(filename):
    data = pd.read_csv(
        filename,
        dtype={
            'item1': str,
            'value1': float,
            'item2': str,
            'value2': float
        })
    data['dif'] = data['value2'] - data['value1']
    print("number of data points: {}".format(len(data)))

    # split training and evaluation data
    d_train, d_eval = train_test_split(data, test_size=0.3)

    # count items
    items = d_train['item1'].append(d_train['item2']).unique().tolist()
    n_items = len(items)
    print("number of unique items: {}".format(n_items))

    return n_items, items, d_train, d_eval


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('dif')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


# PairModel is used to train variables using the paired examples
class PairModel(tf.keras.Model):
    '''
    Train variables using the paired examples

    Tutorial on basic structure:
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model
    '''

    def __init__(self, columns, v0):
        super(PairModel, self).__init__()
        self.input_item_1 = tf.keras.layers.DenseFeatures(columns['item1'])
        self.input_value_1 = tf.keras.layers.DenseFeatures(columns['value1'])
        self.input_item_2 = tf.keras.layers.DenseFeatures(columns['item2'])
        self.input_value_2 = tf.keras.layers.DenseFeatures(columns['value2'])
        self.subtracted = tf.keras.layers.Subtract(name='sub')
        self.dense = tf.keras.layers.Dense(1, use_bias=False)
        self.cen_value = tf.constant(v0, dtype=tf.float32)

    def call(self, inputs):
        # reference part
        hot1_item = self.input_item_1(inputs)
        hot1 = tf.concat([
            hot1_item,
            tf.multiply(hot1_item, self.input_value_1(inputs) - self.cen_value)
        ],
            axis=1)
        # target part
        hot2_item = self.input_item_2(inputs)
        hot2 = tf.concat([
            hot2_item,
            tf.multiply(hot2_item, self.input_value_2(inputs) - self.cen_value)
        ],
            axis=1)
        # subtract the onehot of item2 from onehot of item1
        sub = self.subtracted([hot1, hot2])
        # simple linear regression
        out = self.dense(sub)
        return out


# define model
def create_pair_model(columns, v0):
    '''
    Define pair model
    '''
    model = PairModel(columns, v0)
    model.compile(optimizer=tf.keras.optimizers.Ftrl(0.2),
                  loss='mse',
                  metrics=['mae', 'mse'],
                  run_eagerly=False)
    return model


# make prediction model
class PredModel(tf.Module):
    def __init__(self, item_coefficients, v0):
        '''
        Create lookup tables mapping from items to coefficients
        '''
        # 0-th order coefficients
        self.Table0 = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(item_coefficients['item'], tf.string),
                values=tf.constant(item_coefficients['c0'], tf.float32)),
            default_value=0.0)
        # 1st order coefficients
        self.Table1 = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(item_coefficients['item'], tf.string),
                values=tf.constant(item_coefficients['c1'], tf.float32)),
            default_value=0.0)
        # central value to subtract
        self.cen_value = tf.constant(v0, dtype=tf.float32)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[], dtype=tf.string),
                         tf.TensorSpec(shape=[], dtype=tf.float32)])
    def __call__(self, item, value):
        coeff0 = self.Table0.lookup(item)
        coeff1 = self.Table1.lookup(item)
        new_value = value + coeff0 + coeff1 * (value - self.cen_value)
        return new_value


def run():
    """run"""

    # import data
    n_items, items, d_train, d_eval = import_data(DATA_FILE)

    # build datasets
    ds_train = df_to_dataset(d_train)
    ds_eval = df_to_dataset(d_eval, shuffle=False)

    # columns
    columns = {
        'item1': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                'item1', vocabulary_list=items)),
        'value1': tf.feature_column.numeric_column('value1'),
        'item2': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                'item2', vocabulary_list=items)),
        'value2': tf.feature_column.numeric_column('value2')
    }

    # create model
    pair_model = create_pair_model(columns, CEN_VALUE)

    # train model
    pair_model.fit(ds_train, validation_data=ds_eval, epochs=EPOCHS)

    # print summary
    pair_model.summary()

    # directly get trained variables
    variables = tf.reshape(pair_model.trainable_variables,
                           shape=[2, n_items]).numpy()
    item_coefficients = pd.DataFrame({
        'item': items,
        'c0': variables[0],
        'c1': variables[1]
    })
    item_coefficients.to_csv('item_coefficients.csv', index=False)

    pred_model = PredModel(item_coefficients, CEN_VALUE)
    tf.saved_model.save(pred_model, export_dir='saved_pred_model')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
