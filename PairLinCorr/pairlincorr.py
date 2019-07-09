# -*- coding: utf-8 -*-
"""
Build PairShift model with TensorFlow 2.0
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_FILE = 'input.csv'
SAMPLE_RATIO = 0.01
BATCH_SIZE = 32
EPOCHS = 1


def run():
    """run"""

    #
    # import data
    #
    data = pd.read_csv(
        DATA_FILE, dtype={
            'item1': str,
            'value1': float,
            'item2': str,
            'value2': float
        })
    data['dif'] = data['value2'] - data['value1']

    # sample data for development
    d_trash, d_use = train_test_split(data, test_size=SAMPLE_RATIO)
    print("number of data points: {}".format(len(d_use)))

    # split training and evaluation data
    d_train, d_eval = train_test_split(d_use, test_size=0.3)

    # count items
    items = d_train['item1'].append(d_train['item2']).unique().tolist()
    n_items = len(items)
    print("number of unique items: {}".format(n_items))

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('dif')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    #
    # training and evaluation
    #
    ds_train = df_to_dataset(d_train, batch_size=BATCH_SIZE)
    ds_eval = df_to_dataset(d_eval, shuffle=False, batch_size=2*BATCH_SIZE)

    # columns
    item1_col = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'item1', vocabulary_list=items))
    value1_col = tf.feature_column.numeric_column('value1')
    item2_col = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'item2', vocabulary_list=items))
    value2_col = tf.feature_column.numeric_column('value2')

    # PairModel is used to train variables using the paired examples
    class PairModel(tf.keras.Model):
        '''
        Train variables using the paired examples

        Tutorial on basic structure:
        https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model
        '''

        def __init__(self):
            super(PairModel, self).__init__()
            self.input_item_1 = tf.keras.layers.DenseFeatures(item1_col)
            self.input_value_1 = tf.keras.layers.DenseFeatures(value1_col)
            self.input_item_2 = tf.keras.layers.DenseFeatures(item2_col)
            self.input_value_2 = tf.keras.layers.DenseFeatures(value2_col)
            self.subtracted = tf.keras.layers.Subtract(name='sub')
            self.dense = tf.keras.layers.Dense(1, use_bias=False)

        def call(self, inputs):
            # reference part
            hot1_item = self.input_item_1(inputs)
            hot1 = tf.concat([
                hot1_item,
                tf.multiply(hot1_item,
                            self.input_value_1(inputs))
            ], axis=1)
            # target part
            hot2_item = self.input_item_2(inputs)
            hot2 = tf.concat([
                hot2_item,
                tf.multiply(hot2_item,
                            self.input_value_2(inputs))
            ], axis=1)
            # subtract the onehot of item2 from onehot of item1
            sub = self.subtracted([hot1, hot2])
            # simple linear regression
            out = self.dense(sub)
            return out

    # define model
    def create_pair_model():
        '''
        Define pair model
        '''
        model = PairModel()
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.001),
            loss='mse',
            metrics=['mae', 'mse'],
            run_eagerly=True)
        return model

    # create model
    pair_model = create_pair_model()

    # train model
    pair_model.fit(ds_train, validation_data=ds_eval, epochs=EPOCHS)

    # print summary
    pair_model.summary()

    # directly get trained variables
    variables = tf.reshape(
        pair_model.trainable_variables, shape=[2, n_items]).numpy()
    item_coefficients = pd.DataFrame({
        'item': items,
        'c0': variables[0],
        'c1': variables[1]
    })
    item_coefficients.to_csv('item_coefficients.csv', index=False)

    #
    # make prediction model
    #
    # class PredModel(tf.Module):

    #     def __init__(self):
    #         '''
    #         Create a lookup table mapping from items to coefficients
    #         '''
    #         self.Table = tf.lookup.StaticHashTable(
    #             initializer=tf.lookup.KeyValueTensorInitializer(
    #                 keys=tf.constant(item_coefficients['item'], tf.string),
    #                 values=tf.constant(item_coefficients['coefficient'],
    #                                    tf.float32)),
    #             default_value=0.0)

    #     @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    #     def __call__(self, item):
    #         coeff = self.Table.lookup(item)
    #         return coeff

    # pred_model = PredModel()
    # tf.saved_model.save(pred_model, export_dir='saved_pred_model')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
