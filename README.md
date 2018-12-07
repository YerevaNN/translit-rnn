# translit-rnn: Automatic transliteration with LSTM

This is a tool to transliterate inconsistently romanized text. It is tested on Armenian (`hy-AM`). We invite everyone interested to add more languages. Instructions are below.

[Read more in the corresponding blog post](http://yerevann.github.io/2016/09/09/automatic-transliteration-with-lstm/).

Install required packages:

    pip install -r requirements.txt

Before training on the corpus we need to compute the vocabularies by the following command:

    python make_vocab.py --language hy-AM

The actual training is initiated by a command like this:

    python -u train.py --hdim 1024 --depth 2 --batch_size 200 --seq_len 30 --language hy-AM &> log.txt
    
`--hdim` and `--depth` define biLSTM parameters. `--seq_len` is the maximum length of a character sequence given to the network. The output will be written in `log.txt`.

During the training the models are saved in the `model` folder. The following command will run the test set through the selected model:

    python -u test.py --hdim 1024 --depth 2 --model {MODEL} --language hy-AM

The above command expects that the test set contains text in the _original_ language. The next one takes a file with _romanized_ text and prints the transliterated text:

    python -u test.py --hdim 1024 --depth 2 --model {MODEL} --language hy-AM --translit_path {FILE_NAME}
    
Finally, `plot_loss.py` command will draw the graphs for training and validation losses for the given log file. `--ymax` puts a limit on `y` axis.

    python plot_loss.py --log log.txt --window 10000 --ymax 3
    
    
## Preparing the data for a given language

This is what we did for Armenian. Something similar will be needed for the other not-very-different languages.

First, we prepare the corpus.

1. Download the [Wikipedia dump](https://dumps.wikimedia.org/hywiki/) (e.g. https://dumps.wikimedia.org/hywiki/20160901/hywiki-20160901-pages-articles.xml.bz2) 
2. Extract the dump using [WikiExtractor](https://github.com/attardi/wikiextractor)
3. Remove the remaining tags that (strings starting with '<')
4. Spilt the data three parts (80% - `train.txt`, 10% - `val.txt`, 10% - `test.txt`) and store them in the `languages/LANG_CODE/data/` folder

Next we add some language specific configuration files:

1. Populate the `languages/LANG_CODE/transliteration.json` file with romanization rules, like [this one](https://github.com/YerevaNN/translit-rnn/blob/master/languages/hy-AM/transliteration.json)
2. Populate the `languages/LANG_CODE/long_letters.json` file with an array of the multi-symbol letters of the current language ([Armenian](https://github.com/YerevaNN/translit-rnn/blob/master/languages/hy-AM/long_letters.json) has `ու` and two capitalizations of it: `Ու` and `ՈՒ`)
3. Run `make_vocab.py` to generate the "vocabulary"

