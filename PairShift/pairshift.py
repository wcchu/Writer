# -*- coding: utf-8 -*-
"""
Build PairShift model with TensorFlow 2.0
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_FILE = 'input_100.csv'
BATCH_SIZE = 32
EPOCHS = 1


def run():
    """run"""

    #
    # import data
    #
    data = pd.read_csv(DATA_FILE,
                       dtype={
                           'item1': str,
                           'item2': str,
                           'dif': float
                       })

    # split training and evaluation data
    d_train, d_eval = train_test_split(data, test_size=0.3)

    # count items
    items = d_train['item1'].append(d_train['item2']).unique().tolist()
    print("number of unique items: {}".format(len(items)))

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
    ds_eval = df_to_dataset(d_eval, shuffle=False, batch_size=BATCH_SIZE)

    # columns
    item1_col = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'item1', vocabulary_list=items))
    item2_col = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'item2', vocabulary_list=items))

    # define model class
    class MyModel(tf.keras.Model):
        '''
        Create model class
        https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model
        '''

        def __init__(self):
            super(MyModel, self).__init__()
            self.input1 = tf.keras.layers.DenseFeatures(item1_col)
            self.input2 = tf.keras.layers.DenseFeatures(item2_col)
            self.subtracted = tf.keras.layers.Subtract(name='sub')
            self.dense = tf.keras.layers.Dense(1)

        def call(self, inputs):
            # input1 and input2 take "item1" and "item2" respectively from
            # inputs
            hot1 = self.input1(inputs)
            hot2 = self.input2(inputs)
            # subtract the onehot of item2 from onehot of item1
            sub = self.subtracted([hot1, hot2])
            # simple linear regression
            out = self.dense(sub)
            return out

    model = MyModel()

    # compile model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss='mse',
                  metrics=['mae', 'mse'],
                  run_eagerly=True)

    # train model
    model.fit(ds_train, validation_data=ds_eval, epochs=EPOCHS)

    # print summary
    model.summary()

    #
    # TODO:
    # build a wrapped model which takes one item as input and
    # return item's coefficient
    #

    # save model
    # tf.saved_model.save(model, export_dir='saved_model')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
