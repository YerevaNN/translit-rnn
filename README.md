# translit-rnn
Automatic transliteration with LSTM

python -u make_vocab.py --language hy-AM

python -u train.py --hdim 1024 --depth 2 --batch_size 200 --seq_len 30 --language hy-AM 

python -u test.py --hdim 1024 --depth 2 --model ... --language hy-AM [--translit_path ...] 

python plot_loss.py --log log_the_most_final_2 --window 10000 --ymax 2 
