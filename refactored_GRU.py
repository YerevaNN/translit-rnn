# -*- coding: utf-8 -*

from __future__ import print_function
trans = {
    # capital letters
	u'Ա'  :  {'A' : 1},
	u'Բ'  :  {'B' : 1},
	u'Գ'  :  {'G' : 1},
	u'Դ'  :  {'D' : 1},
	u'Ե'  :  {'E' : 0.8, 'Ye' :  0.1, 'YE': 0.1},
	u'Զ'  :  {'Z' : 1},
	u'Է'  :  {'E' : 1},
	u'Ը'  :  {'@' : 0.45, 'Y' :  0.35 , 'E' :  0.2},
	u'Թ'  :  {'T' : 0.9, 'Th' :  0.05, 'TH' : 0.05},
	u'Ժ'  :  {'J' : 0.5, 'Zh' :  0.15, 'ZH' :  0.15, 'G' :  0.2},
	u'Ի'  :  {'I' :  1},
	u'Լ'  :  {'L' : 1},
	u'Խ'  :  {'X' : 0.7, 'KH' : 0.1, 'Kh' : 0.1, 'GH' : 0.05, 'Gh' : 0.05},
	u'Ծ'  :  {'TS' : 0.3, 'Ts' : 0.3, 'C' : 0.3, '&' : 0.1},
	u'Կ'  :  {'K' : 1},
	u'Հ'  :  {'H' : 1},
	u'Ձ'  :  {'DZ' : 0.4, 'Dz' : 0.4, 'D' : 0.1, 'Z' : 0.1},
	u'Ղ'  :  {'X' : 0.5, 'GH' : 0.25, 'Gh' : 0.25},
	u'Ճ'  :  {'J' : 0.5, 'CH' : 0.2, 'Ch' : 0.2, 'C' : 0.1},
	u'Մ'  :  {'M' : 1},
	u'Յ'  :  {'Y' : 0.9, 'J' : 0.1 },
	u'Ն'  :  {'N' : 1} ,
	u'Շ'  :  {'Sh' : 0.5, 'SH' : 0.5},
	u'Ո'  :  {'O' : 1} , # if
	u'Չ'  :  {'Ch' : 0.5, 'CH' : 0.5},
	u'Պ'  :  {'P' : 1},
	u'Ջ'  :  {'J' : 0.6, 'G' : 0.3, 'DJ': 0.05, 'Dj': 0.05},
	u'Ռ'  :  {'R' : 1},
	u'Ս'  :  {'S' : 1},
	u'Վ'  :  {'V' : 1},
	u'Տ'  :  {'T' : 1},
	u'Ր'  :  {'R' : 1},
	u'Ց'  :  {'C' : 0.8, 'TS' : 0.1, 'Ts' : 0.1},
	u'\u3234'  :  {'U' : 1}, # Ու
	u'\u3235'  :  {'U' : 1}, # ՈՒ
	u'Ւ'  :  {'V' : 1},
	u'Փ'  :  {'P' : 1},
	u'Ք'  :  {'Q' : 0.7, 'K' : 0.3},
	u'Օ'  :  {'O' : 1},
	u'Ֆ'  :  {'F' : 1},
	
	# small letters
	u'ա'  :  {'a' : 1},
	u'բ'  :  {'b' : 1},
	u'գ'  :  {'g' : 1},
	u'դ'  :  {'d' : 1},
	u'ե'  :  {'e' : 0.8, 'ye' : 0.2},
	u'զ'  :  {'z' : 1},
	u'է'  :  {'e' : 1},
	u'ը'  :  {'@' : 0.45, 'y' :  0.35 , 'e' :  0.2},
	u'թ'  :  {'t' :  0.9, 'th' :  0.1},
	u'ժ'  :  {'j' : 0.5, 'zh' :  0.3, 'g' :  0.2},
	u'ի'  :  {'i' :  1},
	u'լ'  :  {'l' : 1},
	u'խ'  :  {'x' : 0.7, 'kh' : 0.2, 'gh' : 0.1},
	u'ծ'  :  {'ts' : 0.6, 'c' : 0.3, '&' : 0.1},
	u'կ'  :  {'k' : 1},
	u'հ'  :  {'h' : 1},
	u'ձ'  :  {'dz' : 0.8, 'd' : 0.1, 'z' : 0.1},
	u'ղ'  :  {'x' : 0.5, 'gh' : 0.5},
	u'ճ'  :  {'j' : 0.5, 'ch' : 0.4, 'c' : 0.1},
	u'մ'  :  {'m' : 1},
	u'յ'  :  {'y' : 0.9, 'j' : 0.1 },
	u'ն'  :  {'n' : 1} ,
	u'շ'  :  {'sh' : 1},
	u'ո'  :  {'o' : 1} , # if
	u'չ'  :  {'ch' : 1},
	u'պ'  :  {'p' : 1},
	u'ջ'  :  {'j' : 0.6, 'g' : 0.3, 'dj': 0.1},
	u'ռ'  :  {'r' : 1},
	u'ս'  :  {'s' : 1},
	u'վ'  :  {'v' : 1},
	u'տ'  :  {'t' : 1},
	u'ր'  :  {'r' : 1},
	u'ց'  :  {'c' : 0.8, 'ts' : 0.2},
	u'\u3233'  :  {'u' : 1},
	u'ւ'  :  {'v' : 1},
	u'փ'  :  {'p' : 1},
	u'ք'  :  {'q' : 0.7, 'k' : 0.3},
	u'և'  :  {'ev' : 0.7, 'yev' : 0.3},
	u'օ'  :  {'o' : 1},
	u'ֆ'  :  {'f' : 1}
}

import numpy as np
import theano
import theano.tensor as T
import lasagne
import urllib2 #For downloading the sample text file. You won't need this if you are providing your own file.
import codecs
import json
import random
import argparse
from datetime import datetime

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
in_text = ' ' + u'\u2001' + u'\u2000' + in_text


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

# Sequence Length
SEQ_LENGTH = 20

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 1024

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 12

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

def gen_data(p, batch_size = BATCH_SIZE, data=in_text, return_target=True):
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
            x[i,ind,trans_to_index[translit[ind]]] = 1
            y[i,ind,char_to_index[armenian[ind]]] = 1
        for ind in range(len(armenian),int(1.3*SEQ_LENGTH)):
            x[i,ind,trans_to_index[u'\u2001']] = 1
            y[i,ind,char_to_index[u'\u2001']] = 1
    return (x,y,p,turned)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdim', default=512, type=int)
    parser.add_argument('--test', default=False)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--seq_len', default=40, type=int)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    
    global BATCH_SIZE, N_HIDDEN, Test, SEQ_LENGTH, NUM_EPOCHS
    SEQ_LENGTH = args.seq_len
    BATCH_SIZE = args.batch_size
    N_HIDDEN = args.hdim
    NUM_EPOCHS = args.num_epochs
    
    Test = (args.test == "True")
    
    print("Building network ...")
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(None, None, trans_vocab_size))

    symbolic_batch_size = lasagne.layers.get_output(l_in).shape[0]
    
    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

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
        
    
    # The output of l_forward_2 of shape (batch_size,seq_len, N_HIDDEN) is then passed through the softmax nonlinearity to 
    # create probability distribution of the prediction
    # The output of this stage is (batch_size, seq_length, vocab_size)
    l_out = lasagne.layers.DenseLayer(sum_layer_1, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.dmatrix('target_output')
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    if not Test:
        global train
        train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
        #compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)
    else:
        global guess
        guess = theano.function([l_in.input_var],network_output,allow_input_downcast=True)
    
    def try_it_out(p,N=5):
        
        sentence_in = ""
        sentence_real = ""
        sentence_out = ""
        for i in range(N):
            x, y = gen_data(p,1)
            tmp = one_hot_matrix_to_sentence(x,translit=True)
            print(tmp.encode('utf-8'))
            sentence_in += tmp.replace('^','')
            tmp = one_hot_matrix_to_sentence(y,translit=False)
            print(tmp.encode('utf-8'))
            sentence_real += tmp.replace('~','')
            tmp = one_hot_matrix_to_sentence(guess(x),translit=False)
            print(tmp.encode('utf-8'))
            sentence_out += tmp.replace('~','')
            p += SEQ_LENGTH
        print(sentence_in)
        print(sentence_real)
        print(sentence_out)
        return p
    
    if args.model:
        f = np.load(args.model)
        param_values = [np.float32(f[i]) for i in range(len(f))]
        lasagne.layers.set_all_param_values(l_out, param_values)
    print("Training ...")
    p = 1
    step_cnt = 0
    avg_cost = 0
    it = 0
    while it < NUM_EPOCHS:
        avg_cost = 0
        date_at_beginning = datetime.now()
        #p = try_it_out(p) # Generate text using the p^th character as the start.

        for _ in range(PRINT_FREQ):
            x,y,p,turned = gen_data(p,BATCH_SIZE)
            if turned:
                it += 1
            avg_cost += train(x, np.reshape(y,(-1,vocab_size)))
        date_after = datetime.now()
        print("Epoch {} average loss = {} Time {} sec.".format(1.0 * it + 1.0 * p / data_size , avg_cost / PRINT_FREQ, (date_after - date_at_beginning).total_seconds()))
        step_cnt += 1
        if step_cnt * BATCH_SIZE > 50000:
            file_name = 'hard_models/only_bidirectional_GRU.' + str(N_HIDDEN) + '.' + str(1.0 * it + 1.0 * p / data_size) + '.epoch.' + str(avg_cost / PRINT_FREQ)  + '.seq_len.' + str(SEQ_LENGTH) + '.loss.' + str(BATCH_SIZE) + '.bs'  + '.npz'
            print("saving to -> " + file_name)
            np.save(file_name, lasagne.layers.get_all_param_values(l_out))
            step_cnt = 0
        
if __name__ == '__main__':
    main()