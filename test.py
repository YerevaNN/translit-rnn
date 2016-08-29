# -*- coding: utf-8 -*

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
import codecs
import editdistance
import random
import argparse
import json

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


#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 50

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



(char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size) = load_vocabulary('aligned_gru')


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


def gen_data(p, batch_size, data, SEQ_LENGTH = 20):
    
    
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

def define_model(N_HIDDEN):
    
    l_in = lasagne.layers.InputLayer(shape=(None, None, trans_vocab_size))

    symbolic_batch_size = lasagne.layers.get_output(l_in).shape[0]
    
    l_forward_1 = lasagne.layers.GRULayer(
        l_in, N_HIDDEN,
        backwards=False)
    
    l_backward_1 = lasagne.layers.GRULayer(
        l_in, N_HIDDEN, 
        backwards=True)
    
    l_reshape_forward_1 = lasagne.layers.ReshapeLayer(l_forward_1, (-1, N_HIDDEN))

    l_forward_1_dense = lasagne.layers.DenseLayer(l_reshape_forward_1, num_units=N_HIDDEN, W = lasagne.init.Normal(), nonlinearity=None)
    
    l_reshape_backward_1 = lasagne.layers.ReshapeLayer(l_backward_1, (-1, N_HIDDEN))
    
    l_backward_1_dense = lasagne.layers.DenseLayer(l_reshape_backward_1, num_units=N_HIDDEN, W = lasagne.init.Normal(), nonlinearity=None)
    
    sum_layer_1 = lasagne.layers.ElemwiseSumLayer(incomings=[l_forward_1_dense,l_backward_1_dense])
    
    l_reshape_sum_1 = lasagne.layers.ReshapeLayer(sum_layer_1, (symbolic_batch_size, -1, N_HIDDEN))
    
    
    l_forward_2 = lasagne.layers.GRULayer(
        l_reshape_sum_1, N_HIDDEN,
        backwards=False)
    
    l_backward_2 = lasagne.layers.GRULayer(
        l_reshape_sum_1, N_HIDDEN,
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

    print("Compiling functions ...")
    guess = theano.function([l_in.input_var],network_output,allow_input_downcast=True)
    
    return(l_out, guess)


def try_it_out(predict, input_file_name, model_name, SEQ_LENGTH = 20):
        
        sentence_in = ""
        sentence_real = ""
        sentence_out = ""
        p = 0
        data = codecs.open(input_file_name, encoding='utf-8').read().replace(u'ու',u'\u3233').replace(u'Ու',u'\u3234').replace(u'ՈՒ',u'\u3235').replace(u'\t',u' ')
        while True: 
            x, y, p, turned = gen_data(p,1,data, SEQ_LENGTH)
            sentence_in += one_hot_matrix_to_sentence(x,translit=True).replace(u'\u2001','').replace(u'\u2000','')
            sentence_real += one_hot_matrix_to_sentence(y,translit=False).replace(u'\u2001','').replace(u'\u2000','')
            sentence_out += one_hot_matrix_to_sentence(predict(x),translit=False).replace(u'\u2001','').replace(u'\u2000','')
            if turned:
                break
        print("Computing editdistance and writing to -> " + 'results.' + model_name.split('/')[-1])
        codecs.open('results.' + model_name.split('/')[-1] ,'w',encoding='utf-8').write(sentence_in + '\n' + 
                                                               sentence_real.replace(u'\u3233',u'ու').replace(u'\u3234',u'Ու').replace(u'\u3235',u'ՈՒ') + '\n' +
                                                               sentence_out.replace(u'\u3233',u'ու').replace(u'\u3234',u'Ու').replace(u'\u3235',u'ՈՒ') + '\n' +
                                                               str(editdistance.eval(sentence_real,sentence_out)) + ' ' + str(len(sentence_real)))


def main(num_epochs=NUM_EPOCHS):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdim', default=512, type=int)
    parser.add_argument('--input', default=None)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--seq_len', default=40, type=int)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
   
    print("Building network ...")
   
    (output_layer, guess) = define_model(N_HIDDEN = args.hdim)
    
    if args.model:
        f = np.load(args.model)
        param_values = [np.float32(f[i]) for i in range(len(f))]
        lasagne.layers.set_all_param_values(output_layer, param_values)
        
    
    
        
    
    
    print("Testing ...")
    
    try_it_out(guess, args.input, args.model, args.seq_len)
    
if __name__ == '__main__':
    main()

