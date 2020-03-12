# Writer

Use RNN with LSTM to generate new text in TensorFlow 2.0.

## Training

The training data (`bible.txt`) is the text of the King James Bible (https://www.kingjamesbibleonline.org/). The first 1000 lines are separately prepared in `bible_1000.txt` for a quick trial.

Run `python learn.py` to build the model and save it in the checkpoint directory. All parameters are defined in the beginning part of the `learn.py` code.

## Writing

Run `python main.py` to deploy the writer locally. Open browser with `localhost:5000` (assuming default port is 5000). Add optional input arguments in the form of `localhost:5000/temp/seed/lmin/lmax` where `temp`, `seed`, `lmin`, `lmax` are temperature, seed text, minimal text length, and maximal text length. If they are not defined in the url, the default values in the beginning part of `main.py` code will be taken.

The model is also deployed to Google App Engine at https://writer-01.appspot.com/. I expect the traffic to be very low so the daily spending limit is set to 1 USD. The access to the app might fail due to this limit.
