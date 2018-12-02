# PairShift

## Building regression model for two symmetric predictor features with TensorFlow

There are N items. A measurement event sends a person to randomly pick two items from N and measure the length difference between the two. After M measurements, we build a table in the form of (item1, item2, difference) in each row, where a "difference" is the length of item2 minus the length of item1. Using this table as the training data, we build a model to predict the "true difference" between any pair of items within these N items.

The prediction should have the following constraints:

1. If item1 == item2, different is 0.
2. If we swap item1 and item2 in an input, the difference should become -difference.

We use TensorFlow to build a linear model with no hidden layer. In order to satisty the above two constraints, each item should have just one weight with opposite signs in item1 and item2, instead of two freely adjustable weights being in item1 and item2. Moreover, the bias is set to 0.

## Anonymize data

The `anonymizer.py` is an helper script to anonymize raw data in the pair form. Use python3 to run the script:

```
python3 anonymizer.py -i raw_data.csv -o anonymized_data.csv
```
