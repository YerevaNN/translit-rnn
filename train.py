# -*- coding: utf-8 -*

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
import codecs
import json
import random
import argparse
from datetime import datetime

trans = json.loads(codecs.open('transliteration.json','r',encoding='utf-8').read())

def isArmenianLetter(s):
    for c in s:
        if ( ( ord(c)< ord(u'ա') or ord(c) > ord(u'և') ) and c != u'\u3233' and c != u'\u3234' and c != u'\u3235' and ( ord(c) < ord(u'Ա') or ord(c) > ord(u'Ֆ') )):
            return False
    return True

def  toTranslit(prevc,c,nextc,trans):
    if not isArmenianLetter(c):
        return c
    if(c == u'ո'):
        if(isArmenianLetter(prevc)):
            return u'o'
        return u'vo'
    if(c == u'Ո'):
        if(isArmenianLetter(prevc)):
            return u'O'
        return u'Vo'
    x = random.random()
    s = 0
    eps = 1e-6
    for i in trans[c]:
        s += trans[c][i]
        if( s > x - eps):
            return i
    print (c,s,"error")

in_text = codecs.open('data/hard_wiki_train',encoding='utf-8').read().replace(u'ու',u'\u3233').replace(u'Ու',u'\u3234').replace(u'ՈՒ',u'\u3235')

#in_text = urllib2.urlopen('https://s3.amazonaws.com/text-datasets/nietzsche.txt').read()
#You can also use your own file
#The file must be a simple text file.
#Simply edit the file name below and uncomment the line.  
#in_text = open('your_file.txt', 'r').read()
#in_text = in_text.encode("utf-8")
in_text = ' \t' + u'\u2001' + u'\u2000' + in_text


#This snippet loads the text file and creates dictionaries to 
#encode characters into a vector-space representation and vice-versa. 

data_size = len(in_text)

def make_vocabulary_files(data, file_name_prefix):
    
    chars = list(set(data))
    char_to_index = { chars[i] : i for i in range(len(chars)) }
    index_to_char = { i : chars[i] for i in range(len(chars)) }
    
    open(file_name_prefix + '.char_to_index.json','w').write(json.dumps(char_to_index))
    open(file_name_prefix + '.index_to_char.json','w').write(json.dumps(index_to_char))
    
    translit = [toTranslit(in_text[i],in_text[i+1],in_text[i+2],trans) for i in range(len(in_text) - 3)]
    trans_chars = list(set(''.join(translit)))
    trans_to_index = { trans_chars[i] : i for i in range(len(trans_chars)) }
    index_to_trans = { i : trans_chars[i] for i in range(len(trans_chars)) }
    trans_vocab_size = len(trans_chars)
    
    open(file_name_prefix + '.trans_to_index.json','w').write(json.dumps(trans_to_index))
    open(file_name_prefix + '.index_to_trans.json','w').write(json.dumps(index_to_trans))

def load_vocabulary(file_name_prefix):
    char_to_index = json.loads(open(file_name_prefix + '.char_to_index.json').read())
    char_to_index = { i : int(char_to_index[i]) for i in char_to_index}
    
    index_to_char = json.loads(open(file_name_prefix + '.index_to_char.json').read())
    index_to_char = { int(i) : index_to_char[i] for i in index_to_char}
    vocab_size = len(char_to_index)
    
    
    trans_to_index = json.loads(open(file_name_prefix + '.trans_to_index.json').read())
    trans_to_index = { i : int(trans_to_index[i]) for i in trans_to_index}
    
    index_to_trans = json.loads(open(file_name_prefix + '.index_to_trans.json').read())
    index_to_trans = { int(i) : index_to_trans[i] for i in index_to_trans}
    trans_vocab_size = len(trans_to_index)
    return (char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size)

#print("Making Vocabulary Files")
#make_vocabulary_files(in_text,'aligned_gru')
print("Loading Vocabulary Files")
(char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size) = load_vocabulary('aligned_gru')

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))


# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 1024



# How often should we check the output?
PRINT_FREQ = 1



# Testing
Test = False

train = None
guess = None

def one_hot_matrix_to_sentence(data, translit = False):
    if data.shape[0] == 1:
        data = data[0]
    sentence = ""
    for i in data:
        if translit:
            sentence += index_to_trans[np.argmax(i)]
        else:
            sentence += index_to_char[np.argmax(i)]
    return sentence

def gen_data(p, SEQ_LENGTH, batch_size = 100, data=in_text):
    x = np.zeros((batch_size,int(1.3*SEQ_LENGTH),trans_vocab_size))
    y = np.zeros((batch_size,int(1.3*SEQ_LENGTH),vocab_size))
    turned = False
    for i in range(batch_size):
        new_p = min(p+SEQ_LENGTH,len(data))
        raw_armenian = data[p:new_p]
        if new_p != len(data):
            if max([raw_armenian.rfind(u' '),raw_armenian.rfind(u'\t'),raw_armenian.rfind(u'\n')]) > 0:
                new_p = max([raw_armenian.rfind(u' '),raw_armenian.rfind(u'\t'),raw_armenian.rfind(u'\n')]) 
                raw_armenian = ' ' + raw_armenian[:new_p] + ' '
                p += new_p
            else:
                p = new_p
                raw_armenian = ' ' + raw_armenian + ' '
        else:
            raw_armenian = ' ' + raw_armenian + ' '
            p = 0
            turned = True
        armenian = []
        translit = []
        for ind in range(1,len(raw_armenian)-1):
            trans_char = toTranslit(raw_armenian[ind-1], raw_armenian[ind], raw_armenian[ind+1], trans)
            translit.append(trans_char[0])
            if len(trans_char) > 1:
                armenian.append(u'\u2000')
                translit.append(trans_char[1])
            armenian.append(raw_armenian[ind])
        for ind in range(len(armenian)):
            y[i,ind,char_to_index[armenian[ind]]] = 1
            x[i,ind,trans_to_index[translit[ind]]] = 1
        for ind in range(len(armenian),int(1.3*SEQ_LENGTH)):
            x[i,ind,trans_to_index[u'\u2001']] = 1
            y[i,ind,char_to_index[u'\u2001']] = 1
    return (x,y,p,turned)

def define_model(N_HIDDEN , LEARNING_RATE,  GRAD_CLIP):
    
    l_in = lasagne.layers.InputLayer(shape=(None, None, trans_vocab_size))

    symbolic_batch_size = lasagne.layers.get_output(l_in).shape[0]
    
    l_forward_1 = lasagne.layers.GRULayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        backwards=False)
    
    l_backward_1 = lasagne.layers.GRULayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        backwards=True)
    
    l_reshape_forward_1 = lasagne.layers.ReshapeLayer(l_forward_1, (-1, N_HIDDEN))

    l_forward_1_dense = lasagne.layers.DenseLayer(l_reshape_forward_1, num_units=N_HIDDEN, W = lasagne.init.Normal(), nonlinearity=None)
    
    l_reshape_backward_1 = lasagne.layers.ReshapeLayer(l_backward_1, (-1, N_HIDDEN))
    
    l_backward_1_dense = lasagne.layers.DenseLayer(l_reshape_backward_1, num_units=N_HIDDEN, W = lasagne.init.Normal(), nonlinearity=None)
    
    sum_layer_1 = lasagne.layers.ElemwiseSumLayer(incomings=[l_forward_1_dense,l_backward_1_dense])
    
    l_reshape_sum_1 = lasagne.layers.ReshapeLayer(sum_layer_1, (symbolic_batch_size, -1, N_HIDDEN))
    
    
    l_forward_2 = lasagne.layers.GRULayer(
        l_reshape_sum_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        backwards=False)
    
    l_backward_2 = lasagne.layers.GRULayer(
        l_reshape_sum_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        backwards=True)
    
    l_reshape_forward_2 = lasagne.layers.ReshapeLayer(l_forward_2, (-1, N_HIDDEN))

    l_forward_2_dense = lasagne.layers.DenseLayer(l_reshape_forward_2, num_units=N_HIDDEN, W = lasagne.init.Normal(), nonlinearity=None)
    
    l_reshape_backward_2 = lasagne.layers.ReshapeLayer(l_backward_2, (-1, N_HIDDEN))
    
    l_backward_2_dense = lasagne.layers.DenseLayer(l_reshape_backward_2, num_units=N_HIDDEN, W = lasagne.init.Normal(), nonlinearity=None)
    
    sum_layer_2 = lasagne.layers.ElemwiseSumLayer(incomings=[l_forward_2_dense,l_backward_2_dense])
    
    l_out = lasagne.layers.DenseLayer(sum_layer_2, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.dmatrix('target_output')
    
    network_output = lasagne.layers.get_output(l_out)

    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    
    return(l_out,train)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdim', default=512, type=int)
    parser.add_argument('--grad_clip', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--seq_len', default=60, type=int)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
   
    print("Building network ...")
   
    (output_layer, train) = define_model(args.hdim, args.lr, args.grad_clip)
    
    if args.model:
        f = np.load(args.model)
        param_values = [np.float32(f[i]) for i in range(len(f))]
        lasagne.layers.set_all_param_values(output_layer, param_values)
        
    
    print("Training ...")
    p = 1
    step_cnt = 0
    avg_cost = 0
    it = 0
    while it < args.num_epochs:
        avg_cost = 0
        date_at_beginning = datetime.now()

        for _ in range(PRINT_FREQ):
            x,y,p,turned = gen_data(p,args.seq_len, args.batch_size)
            if turned:
                it += 1
            avg_cost += train(x, np.reshape(y,(-1,vocab_size)))
        date_after = datetime.now()
        print("Epoch {} average loss = {} Time {} sec.".format(1.0 * it + 1.0 * p / data_size , avg_cost / PRINT_FREQ, (date_after - date_at_beginning).total_seconds()))
        
        step_cnt += 1
        if step_cnt * args.batch_size > 50000:
            file_name = 'models/only_bidirectional_GRU.' + str(args.hdim) + '.' + str(1.0 * it + 1.0 * p / data_size) + '.epoch.' + str(avg_cost / PRINT_FREQ)  + '.loss.' + str(args.seq_len) + '.seq_len.' + str(args.batch_size) + '.bs'  + '.npz'
            print("saving to -> " + file_name)
            np.save(file_name, lasagne.layers.get_all_param_values(output_layer))
            step_cnt = 0
        
if __name__ == '__main__':
    main()