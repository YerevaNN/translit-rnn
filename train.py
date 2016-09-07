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


VALIDATION_DATA_PATH = 'data/unfiltered_sartre'
TRAIN_DATA_PATH = 'data/hard_wiki_train'


long_letters = json.loads(codecs.open('long_letters.json','r',encoding='utf-8').read())
long_letter_mapping = { long_letters[i] : unichr(ord(u'\u2002') + i) for i in range(len(long_letters)) }
trans = json.loads(codecs.open('new_trans.json','r',encoding='utf-8').read())
tmp_trans = trans.copy()

for c in tmp_trans:
    if c in long_letters:
        trans[long_letter_mapping[c]] = trans[c]

del tmp_trans
train_text = codecs.open(TRAIN_DATA_PATH, encoding='utf-8').read()
val_text = codecs.open(VALIDATION_DATA_PATH, encoding='utf-8').read()

for letter in long_letter_mapping:
    train_text = train_text.replace(letter,long_letter_mapping[letter])
    val_text = val_text.replace(letter,long_letter_mapping[letter])
    
train_text = ' ' + u'\u2001' + u'\u2000' + train_text

data_size = len(train_text)

#Lasagne Seed for Reproducibility

lasagne.random.set_rng(np.random.RandomState(1))


# How often information is printed to log

PRINT_FREQ = 1


### Replaces all non given characters with '#'

def valid(sequence):
    
    valids = [u'\u2000', u'\u2001',';',':','-',',',' ','\n','\t']  + \
             [chr(ord('0') + i) for i in range(10)] + list(set(''.join([''.join([s for s in trans[c]]) for c in trans])))
    
    ans = []
    
    for c in sequence:
        if c in valids:
            ans.append(c)
        else:
            ans.append('#')
    return ans

    
### Checks if a character is from a native languge

def isNativeLetter(s):
    for c in s:
        if c not in trans:
            return False
    return True


### Translates one character to translit, given probabilistic mapping. 
### Previous and next characters are given for some armenian-specific parts

def  toTranslit(prevc,c,nextc,trans):
    if not isNativeLetter(c):
        return c
    ### Armenian Specific Snippet
    
    if(c == u'ո'):
        if(isNativeLetter(prevc)):
            return u'o'
        return u'vo'
    if(c == u'Ո'):
        if(isNativeLetter(prevc)):
            return u'O'
        return u'Vo'

    ###
    x = random.random()
    s = 0
    eps = 1e-6
    for i in trans[c]:
        s += trans[c][i]
        if( s > x - eps):
            return i
    print (c,s,"error")

### Makes jsons for future mapping of letters to indices and vice versa

def make_vocabulary_files(data, file_name_prefix):
    pointer = 0
    done = False
    s_l = 100000
    chars = set()
    trans_chars = set()
    while not done:
        new_p = min(pointer + s_l ,len(data))
        raw_native = data[pointer : new_p]
        if new_p != len(data):
            pointer = new_p
            raw_native = ' ' + raw_native + ' '
        else:
            raw_native = ' ' + raw_native + ' '
            done = True
        native = []
        translit = []
        for ind in range(1,len(raw_native)-1):
            trans_char = toTranslit(raw_native[ind-1], raw_native[ind], raw_native[ind+1], trans)
            translit.append(trans_char[0])
            if len(trans_char) > 1:
                native.append(u'\u2000')
                translit.append(trans_char[1])
            native.append(raw_native[ind])
        translit = valid(translit)
        for i in range(len(native)):
            if translit[i] == '#':
                native[i] = '#'
        chars = chars.union(set(native))
        trans_chars = trans_chars.union(set(translit))
        print(str(100.0*pointer/len(data)) + "% done       ", end='\r')
        
    chars = list(chars)
    char_to_index = { chars[i] : i for i in range(len(chars)) }
    index_to_char = { i : chars[i] for i in range(len(chars)) }
    
    open(file_name_prefix + '.char_to_index.json','w').write(json.dumps(char_to_index))
    open(file_name_prefix + '.index_to_char.json','w').write(json.dumps(index_to_char))
    
    trans_chars = list(trans_chars)
    trans_to_index = { trans_chars[i] : i for i in range(len(trans_chars)) }
    index_to_trans = { i : trans_chars[i] for i in range(len(trans_chars)) }
    trans_vocab_size = len(trans_chars)
    
    open(file_name_prefix + '.trans_to_index.json','w').write(json.dumps(trans_to_index))
    open(file_name_prefix + '.index_to_trans.json','w').write(json.dumps(index_to_trans))

### Loads vocabulary mappings from specified json files

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

print("Making Vocabulary Files")
make_vocabulary_files(train_text,'small')

print("Loading Vocabulary Files")
(char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size) = load_vocabulary('small')

### Converts one sequence of one hot vectors to a string sentence

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

### Generates training examples from data, starting from given index
### and returns the index where it stopped
### also returns the number of sequences skipped (because of lack of native characters)
### and a boolean showing whether generation passed one iteration over data or not

def gen_data(p, seq_len, batch_size = 100, data=train_text):
    
    samples = []
    batch_seq_len = 0
    non_native_sequences = 0
    turned = False
    
    for i in range(batch_size):
        while True:
            new_p = min(p+seq_len,len(data))
            raw_native = data[p:new_p]
            if new_p != len(data):
                if max([raw_native.rfind(u' '),raw_native.rfind(u'\t'),raw_native.rfind(u'\n')]) > 0:
                    new_p = max([raw_native.rfind(u' '),raw_native.rfind(u'\t'),raw_native.rfind(u'\n')]) 
                    raw_native = ' ' + raw_native[:new_p] + ' '
                    p += new_p
                else:
                    p = new_p
                    raw_native = ' ' + raw_native + ' '
            else:
                raw_native = ' ' + raw_native + ' '
                p = 0
                turned = True
            native_letter_count = sum([1 for c in raw_native if isNativeLetter(c)])
            if native_letter_count * 3 > len(raw_native):
                break
            else:
                non_native_sequences += 1

        native = []
        translit = []
        for ind in range(1,len(raw_native)-1):
            trans_char = toTranslit(raw_native[ind-1], raw_native[ind], raw_native[ind+1], trans)
            translit.append(trans_char[0])
            trans_ind = 1
            while len(trans_char) > trans_ind:
                native.append(u'\u2000')
                translit.append(trans_char[trans_ind])
                trans_ind += 1
            native.append(raw_native[ind])
            
        translit = valid(translit)
        for ind in range(len(native)):
            if translit[ind] == '#':
                native[ind] = '#' 
        
        x = np.zeros((len(native), trans_vocab_size))
        y = np.zeros((len(native), vocab_size))
        for ind in range(len(native)):
            x[ind,trans_to_index[translit[ind]]] = 1
            y[ind,char_to_index[native[ind]]] = 1
        
        batch_seq_len = max(batch_seq_len, len(native))
        samples.append((x,y))
        
    x = np.zeros((batch_size, batch_seq_len, trans_vocab_size))
    y = np.zeros((batch_size, batch_seq_len, vocab_size))
    
    for i in range(batch_size):
        x[i, : len(samples[i][0]), :] = samples[i][0]
        y[i, : len(samples[i][1]), :] = samples[i][1]
        for j in range(len(samples[i][0]), batch_seq_len):
            x[i, j, trans_to_index[u'\u2001']] = 1
            y[i, j, char_to_index[u'\u2001']] = 1
    
    return (x,y,p,turned,non_native_sequences)
    
### Defines lasagne model
### Returns output layer and theano functions for training and computing the cost

def define_model(N_HIDDEN, depth, LEARNING_RATE,  GRAD_CLIP):
    
    l_input = lasagne.layers.InputLayer(shape=(None, None, trans_vocab_size))
    network = l_input
    symbolic_batch_size = lasagne.layers.get_output(network).shape[0]
    
    while depth > 0 :
        
        l_forward = lasagne.layers.LSTMLayer(
            network, N_HIDDEN, grad_clipping=GRAD_CLIP,
            backwards=False)
        
        l_backward = lasagne.layers.LSTMLayer(
            network, N_HIDDEN, grad_clipping=GRAD_CLIP,
            backwards=True)
        
        network = lasagne.layers.ConcatLayer(incomings=[l_forward,l_backward], axis = 2)
        network = lasagne.layers.ReshapeLayer(network, (-1, 2*N_HIDDEN))
        network = lasagne.layers.DenseLayer(network, num_units=N_HIDDEN, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.tanh)
        network = lasagne.layers.ReshapeLayer(network, (symbolic_batch_size, -1, N_HIDDEN))
        
        depth -= 1
    
    network = lasagne.layers.ReshapeLayer(network, (-1, N_HIDDEN) )
    l_input_reshape = lasagne.layers.ReshapeLayer(l_input, (-1, trans_vocab_size))
    network = lasagne.layers.ConcatLayer(incomings=[network,l_input_reshape], axis = 1)
    
    l_out = lasagne.layers.DenseLayer(network, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.dmatrix('target_output')
    
    network_output = lasagne.layers.get_output(l_out)

    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    print("Computing Updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    
    print("Compiling Functions ...")
    compute_cost = theano.function([l_input.input_var, target_values], cost, allow_input_downcast=True)
    train = theano.function([l_input.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    
    return(l_out,train, compute_cost)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdim', default=512, type=int)
    parser.add_argument('--grad_clip', default=None, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--seq_len', default=60, type=int)
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--model', default=None)
    parser.add_argument('--start_from', default=0, type=float)
    args = parser.parse_args()
   
    print("Building Network ...")
   
    (output_layer, train, cost) = define_model(args.hdim, args.depth, args.lr, args.grad_clip)
    
    if args.model:
        f = np.load(args.model)
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
            x,y,p,turned, non_native_sequences = gen_data(p,args.seq_len, args.batch_size)
            if turned:
                it += 1
            avg_cost += train(x, np.reshape(y,(-1,vocab_size)))
            non_native_skipped += non_native_sequences
        date_after = datetime.now()
        print("Epoch {} average loss = {} Time {} sec. Nonnatives skipped {}".format(1.0 * it + 1.0 * p / data_size , avg_cost / PRINT_FREQ, (date_after - date_at_beginning).total_seconds(), non_native_skipped))
        
        step_cnt += 1
        if step_cnt * args.batch_size > 20000:
            print('computing validation loss...')
            val_turned = False
            val_p = 0
            val_steps = 0.
            val_cost = 0.
            while not val_turned:
                x, y, val_p, val_turned, non_native = gen_data(val_p,
                        args.seq_len, args.batch_size, data=val_text)
                val_steps += 1
                val_cost += cost(x,np.reshape(y,(-1,vocab_size)))
            print('validation loss is ' + str(val_cost/val_steps))
            file_name = 'models/final_bidirectional_LSTMs.' + str(args.hdim) + '.hdim.' + str(args.depth) + '.depth.' + str(1.0 * it + 1.0 * p / data_size) + '.epoch.' + str(avg_cost / PRINT_FREQ)  + '.loss.' + str(args.seq_len) + '.seq_len.' + str(args.batch_size) + '.bs'  + '.npz'
            print("saving to -> " + file_name)
            np.save(file_name, lasagne.layers.get_all_param_values(output_layer))
            step_cnt = 0
        
if __name__ == '__main__':
    main()
