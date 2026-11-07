# Writer

Use RNN with LSTM to generate new text in TensorFlow 2.

## Training

I prepared 3 training datasets in `data/`:

1. `data/bible.txt` - The text of the King James Bible (https://www.kingjamesbibleonline.org/).
2. `data/trump.txt` - The tweets by Donald Trump until 2021-01-08 15:44:28 (https://www.thetrumparchive.com/). I removed the emojis in the way suggested in https://stackoverflow.com/a/44905730.
3. `data/shakespeare.txt` - The complete works of William Shakespeare (https://www.gutenberg.org/ebooks/100).

Use `DATA_DIR` in `learn.py` to choose the training dataset. Run `python learn.py` to build the model and save it in the checkpoint directory. All parameters are defined in the beginning part of the `learn.py` code.

## Writing

Run `python main.py` to start the writer locally. Open a browser at `localhost:5000` (assuming the default port is 5000). Add optional input arguments in the form of `localhost:5000/temp/seed/lmin/lmax` where `temp`, `seed`, `lmin`, and `lmax` are the temperature, seed text, minimum text length, and maximum text length. If they are not defined in the URL, the default values at the beginning of `main.py` will be used.
