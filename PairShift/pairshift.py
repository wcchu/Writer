import numpy as np
import pandas as pd
import tensorflow as tf
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input-file",
    dest="input_file",
    help="read data from FILE",
    metavar="FILE")
parser.add_argument(
    "-o",
    "--output-file",
    dest="output_file",
    help="write data to FILE",
    metavar="FILE")
args = parser.parse_args()

data = pd.read_csv(
    args.input_file, dtype={
        'item1': str,
        'item2': str,
        'dif': float
    })
x = data[['item1', 'item2']]
y = data['dif']

# count feature values
list_items = x['item1'].append(x['item2']).drop_duplicates().tolist()
n_items = len(list_items)

# columns
item1_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'item1', vocabulary_list=list_items)
item2_col = tf.feature_column.categorical_column_with_vocabulary_list(
    'item2', vocabulary_list=list_items)
feat_cols = [item1_col, item2_col]

# estimator
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# train model
estimator.train(
    input_fn=tf.estimator.inputs.pandas_input_fn(
        x=x, y=y, batch_size=200, num_epochs=5, shuffle=True))

# construct a table for prediction
x_pred = pd.DataFrame(columns=x.columns)
x_pred = x_pred.append(x[['item1']].drop_duplicates(), ignore_index=True)
x_pred = x_pred.append(x[['item2']].drop_duplicates(), ignore_index=True)
x_pred = x_pred.replace(np.nan, '', regex=True)
x_pred.loc[len(x_pred)] = ['', '']

# predict for each item
predictions = list(
    estimator.predict(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=x_pred, batch_size=len(x_pred), num_epochs=1, shuffle=False)))

# output
pred = []
for i in predictions:
    pred.append(np.float64(i['predictions'][0]))
x_pred['shift'] = pred
x_pred.to_csv(args.output_file, sep=',', index=False)
