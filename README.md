# translit-rnn: Automatic transliteration with LSTM

This is a tool to transliterate inconsistently romanized text. It is tested on Armenian (`hy-AM`). We invite everyone interested to add more languages. Instructions are below.

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
    
    
Prepairing the data

1. Download wikipedia dump (https://dumps.wikimedia.org/hywiki/20160901/hywiki-20160901-pages-articles.xml.bz2) 
2. Extract data using this code (https://github.com/attardi/wikiextractor)
3. Remove remaining tags (string starting with '<' )
4. Spilt into three parts (0.8 - train, 0.1 - test, 0.1 - validation)


