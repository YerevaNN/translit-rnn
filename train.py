# -*- coding: utf-8 -*

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
import codecs
import json
import argparse
import random
import utils
from datetime import datetime

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdim', default=512, type=int)
    parser.add_argument('--grad_clip', default=100, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
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
    step_cnt = 0
    date_at_beginning = datetime.now()
    last_time = date_at_beginning
    for epoch in range(args.num_epochs):
        train_text = train_text.split(u'։')
        random.shuffle(train_text)
        train_text = u'։'.join(train_text)
        avg_cost = 0.0
        count = 0
        num_of_samples = 0
        num_of_chars = 0
        for (x, y) in utils.data_generator(train_text, args.seq_len, args.batch_size, trans, trans_to_index, char_to_index, is_train = True):
            sample_cost = train(x, np.reshape(y,(-1,vocab_size)))
            sample_cost = float(sample_cost)
            count += 1
            num_of_samples += x.shape[0]
            num_of_chars += x.shape[0] * x.shape[1]
            
            time_now = datetime.now()
            if (time_now - last_time).total_seconds() > 60 * 1: # 10 minutes
                print('Computing validation loss...')
                val_cost = 0.0
                val_count = 0.0
                for ((x_val, y_val, indices, delimiters), non_valids_list) in utils.data_generator(val_text, args.seq_len, args.batch_size, trans, trans_to_index, char_to_index, is_train = False):
                    val_cost += x_val.shape[0] *cost(x_val,np.reshape(y_val,(-1,vocab_size)))
                    val_count += x_val.shape[0]
                print('Validation loss is {}'.format(val_cost/val_count))
                
                file_name = 'languages/{}/models/{}.hdim{}.depth{}.seq_len{}.bs{}.time{:4f}.epoch{}.loss{:.4f}'.format(args.language, args.model_name_prefix, args.hdim, args.depth, args.seq_len, args.batch_size, (time_now - date_at_beginning).total_seconds()/60, epoch, val_cost/val_count)
                print("saving to -> " + file_name)
                np.save(file_name, lasagne.layers.get_all_param_values(output_layer))
                last_time = datetime.now()
            
            print("On step #{} loss is {:.4f}, samples passed {}, chars_passed {}, {:.4f}% of an epoch {} time passed {:4f}"\
                  .format(count, sample_cost, num_of_samples, num_of_chars, 100.0*num_of_chars/len(train_text), epoch, (time_now - date_at_beginning).total_seconds()/60.0))
                  
            avg_cost += sample_cost
	date_after = datetime.now()
        print("After epoch {} average loss is {:.4f} Time {} sec.".format( epoch , avg_cost/count, (date_after - date_at_beginning).total_seconds()))

        

        
if __name__ == '__main__':
    main()
