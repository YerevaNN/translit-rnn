# -*- coding: utf-8 -*

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
import codecs
import json
import argparse
import utils
from datetime import datetime

PRINT_FREQ = 1


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdim', default=512, type=int)
    parser.add_argument('--grad_clip', default=100, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--seq_len', default=60, type=int)
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--model', default=None)
    parser.add_argument('--model_name_prefix', default='model')
    parser.add_argument('--language', default='hy-AM')
    parser.add_argument('--start_from', default=0, type=float)
    args = parser.parse_args()
   
    print("Loading Files")
    
    (char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size) = utils.load_vocabulary(language = args.language)
    (train_text, val_text, trans) = utils.load_language_data(language = args.language)
    data_size = len(train_text)
    
    print("Building Network ...")
   
    (output_layer, train, cost) = utils.define_model(args.hdim, args.depth, args.lr, args.grad_clip, trans_vocab_size, vocab_size, is_train = True)
    
    if args.model:
        f = np.load('languages/' + args.language + '/models/' + args.model)
        param_values = [np.float32(f[i]) for i in range(len(f))]
        lasagne.layers.set_all_param_values(output_layer, param_values)
    
    print("Training ...")
    p = int(len(train_text) * args.start_from) + 1
    step_cnt = 0
    avg_cost = 0
    it = 0
    while it < args.num_epochs:
        avg_cost = 0
        date_at_beginning = datetime.now()
        non_native_skipped = 0
        for _ in range(PRINT_FREQ):
            x,y,p,turned, non_native_sequences = utils.gen_data(p,args.seq_len, args.batch_size, train_text, trans, trans_to_index, char_to_index)
            if turned:
                it += 1
            avg_cost += train(x, np.reshape(y,(-1,vocab_size)))
            non_native_skipped += non_native_sequences
        date_after = datetime.now()
        print("Epoch {} average loss = {} Time {} sec. Nonnatives skipped {}".format(1.0 * it + 1.0 * p / data_size , avg_cost / PRINT_FREQ, (date_after - date_at_beginning).total_seconds(), non_native_skipped))
        
        step_cnt += 1
        if True: #step_cnt * args.batch_size > 100000:
            print('computing validation loss...')
            val_turned = False
            val_p = 0
            val_steps = 0.
            val_cost = 0.
            while not val_turned:
                x, y, val_p, val_turned, non_native = utils.gen_data(val_p,args.seq_len, args.batch_size, val_text, trans, trans_to_index, char_to_index)
                val_steps += 1
                val_cost += cost(x,np.reshape(y,(-1,vocab_size)))
            print('validation loss is ' + str(val_cost/val_steps))
            file_name = 'languages/' + args.language + '/models/' + args.model_name_prefix  +  '.hdim' + str(args.hdim) + '.depth' + str(args.depth) + '.seq_len' + str(args.seq_len) + '.bs' + str(args.batch_size) + '.epoch' + str(1.0 * it + 1.0 * p / data_size) + '.loss' + str(avg_cost / PRINT_FREQ) + '.npz'
            print("saving to -> " + file_name)
            np.save(file_name, lasagne.layers.get_all_param_values(output_layer))
            step_cnt = 0
        
if __name__ == '__main__':
    main()
