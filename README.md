# Writer

Use RNN with LSTM to generate new text in TensorFlow 2.

## Training

The dataset is composed of data from two different sources:

1. The text of the King James Bible (https://www.kingjamesbibleonline.org/).
2. The tweets by Donald Trump until 2021-01-08 15:44:28 (https://www.thetrumparchive.com/). I removed the emojis in the way suggested in https://stackoverflow.com/a/44905730.

In the data file `data/data.txt`, the first 30382 lines are from source 1, while line 30382 to line 89110 are from source 2. The first 1000 lines of this data file are separately prepared in `data/data_1000.txt` for a quick trial.

Run `python learn.py` to build the model and save it in the checkpoint directory. All parameters are defined in the beginning part of the `learn.py` code.

## Writing

Run `python main.py` to deploy the writer locally. Open browser with `localhost:5000` (assuming default port is 5000). Add optional input arguments in the form of `localhost:5000/temp/seed/lmin/lmax` where `temp`, `seed`, `lmin`, `lmax` are temperature, seed text, minimal text length, and maximal text length. If they are not defined in the url, the default values in the beginning part of `main.py` code will be taken. I also deployed the app through google app engine to https://writer-01.ey.r.appspot.com. I expect the traffic to be very low so the daily spending limit is set to 1 USD. The access to the app might fail due to this limit.
