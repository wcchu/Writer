# Writer

Use RNN with LSTM to generate new text in TensorFlow 2.

## Training

I prepared 3 training datasets in `data/`:

1. `data/bible.txt` - The text of the King James Bible (https://www.kingjamesbibleonline.org/).
2. `data/trump.txt` - The tweets by Donald Trump until 2021-01-08 15:44:28 (https://www.thetrumparchive.com/). I removed the emojis in the way suggested in https://stackoverflow.com/a/44905730.
3. `data/les_miserables.txt` - Les Misérables by Victor Hugo, translated to English by Isabel Florence Hapgood (https://www.gutenberg.org/ebooks/135).

Use `DATA_DIR` in `learn.py` to choose the training dataset. Run `python learn.py` to build the model and save it in the checkpoint directory. All parameters are defined in the beginning part of the `learn.py` code.

## Writing

Run `python main.py` to deploy the writer locally. Open browser with `localhost:5000` (assuming default port is 5000). Add optional input arguments in the form of `localhost:5000/temp/seed/lmin/lmax` where `temp`, `seed`, `lmin`, `lmax` are temperature, seed text, minimal text length, and maximal text length. If they are not defined in the url, the default values in the beginning part of `main.py` code will be taken. I also deployed the app through google app engine to https://writer-01.ey.r.appspot.com. I expect the traffic to be very low so the daily spending limit is set to 1 USD. The access to the app might fail due to this limit.
