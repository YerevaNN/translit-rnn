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

def valid(s):
    valids = ['@',';',':','-',',',' ','\n','\t','&'] + [chr( ord('a') + i) for i in range(26)] + [chr( ord('A') + i) for i in range(26)]
    ans = []
    non_valids = []
    for c in s:
        if c in valids:
            ans.append(c)
        else:
            ans.append('#')
            non_valids.append(c)
    return (ans,non_valids)

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



(char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size) = load_vocabulary('small')


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

def gen_validation_data(p,data,seq_len):
    
    x = np.zeros((1,int(1.3*seq_len),trans_vocab_size))
    turned = False
    new_p = min(p+seq_len,len(data))
    raw_translit = data[p:new_p]
    
    if new_p != len(data):
        if max([raw_translit.rfind(u' '),raw_translit.rfind(u'\t'),raw_translit.rfind(u'\n')]) > 0:
            new_p = max([raw_translit.rfind(u' '),raw_translit.rfind(u'\t'),raw_translit.rfind(u'\n')])
            raw_translit = raw_translit[:new_p]
            p += new_p
        else:
            p = new_p
    else:
        p = 0
        turned = True
    (translit,non_valids) = valid([trans for trans in raw_translit])
    for ind in range(len(translit)):
        x[0,ind,trans_to_index[translit[ind]]] = 1
    for ind in range(len(translit),int(1.3*seq_len)):
        x[0,ind,trans_to_index[u'\u2001']] = 1
    
    return (x,non_valids,p,turned)    
    
def gen_data(p, batch_size, data, SEQ_LENGTH):
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
        
        (translit,non_valids) = valid(translit)
        for ind in range(len(armenian)):
            if translit[ind] == '#':
                armenian[ind] = '#' 
        
        for ind in range(len(armenian)):
            x[i,ind,trans_to_index[translit[ind]]] = 1
            y[i,ind,char_to_index[armenian[ind]]] = 1
        for ind in range(len(armenian),int(1.3*SEQ_LENGTH)):
            x[i,ind,trans_to_index[u'\u2001']] = 1
            y[i,ind,char_to_index[u'\u2001']] = 1
            
    return (x,y,non_valids,p,turned)
def get_residual_weight_matrix(network,csv_name):
    W = network.get_params()[0].get_value()[-63:,:]
    fr = ['" "'] + ['"' + index_to_char[i] + '"' for i in range(len(index_to_char))]
    rows = [[index_to_trans[i]] + [x for x in W[i] ] for i in range(len(index_to_trans))]
    print(rows)
    codecs.open(csv_name,'w',encoding='utf-8').write(','.join(fr) + '\n' + '\n'.join(['"' + row[0] + '",' + ','.join([ "%.3f" %(r) for r in row[1:] ]) for row in rows]))
def define_model(N_HIDDEN, depth):
    
    l_input = lasagne.layers.InputLayer(shape=(None, None, trans_vocab_size))
    network = l_input
    symbolic_batch_size = lasagne.layers.get_output(network).shape[0]
    
    while depth > 0 :
        
        l_forward = lasagne.layers.LSTMLayer(
            network, N_HIDDEN, 
            backwards=False)
        
        l_backward = lasagne.layers.LSTMLayer(
            network, N_HIDDEN,
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

    print("Compiling functions ...")
    guess = theano.function([l_input.input_var],network_output,allow_input_downcast=True)
    
    return(l_out, guess)

def translate_translit(predict, trans_file_name , seq_len):
    data = codecs.open(trans_file_name, encoding='utf-8').read()
    p = 0
    turned = False
    sentence_out = "\n"
    while not turned:
        x, non_valids, p, turned = gen_validation_data(p, data, seq_len)
        guess = one_hot_matrix_to_sentence(predict(x),translit=False).replace(u'\u2001','').replace(u'\u2000','').replace(u'\u3233',u'ու').replace(u'\u3234',u'Ու').replace(u'\u3235',u'ՈՒ')
        final_guess = ""
        ind = 0
        for c in guess:
            if c == '#' and ind < len(non_valids):
                final_guess += non_valids[ind]
                ind += 1
            else:
                final_guess += c
        sentence_out += final_guess
        print(str(100.0*p/len(data)) + "% done       ", end='\r')
    print(sentence_out)

def try_it_out(predict, input_file_name, model_name, SEQ_LENGTH):
        
    sentence_in = ""
    sentence_real = ""
    sentence_out = ""
    p = 0
    turned = False
    data = codecs.open(input_file_name, encoding='utf-8').read().replace(u'ու',u'\u3233').replace(u'Ու',u'\u3234').replace(u'ՈՒ',u'\u3235').replace(u'\t',u' ')
    while not turned: 
        x, y, non_valids, p, turned = gen_data(p,1,data, SEQ_LENGTH)
        sentence_in += one_hot_matrix_to_sentence(x,translit=True).replace(u'\u2001','').replace(u'\u2000','')
        real_without_signs = one_hot_matrix_to_sentence(y,translit=False).replace(u'\u2001','').replace(u'\u2000','')
        ind = 0
        real = ""
        for c in real_without_signs:
            if c == '#' and ind < len(non_valids):
                real += non_valids[ind]
                ind += 1
            else:
                real += c
        sentence_real += real
        guess = one_hot_matrix_to_sentence(predict(x),translit=False).replace(u'\u2001','').replace(u'\u2000','')
        ind = 0
        final_guess = ""
        for c in guess:
            if c == '#' and ind < len(non_valids):
                final_guess += non_valids[ind]
                ind += 1
            else:
                final_guess += c
        sentence_out += final_guess
        print(str(100.0*p/len(data)) + "% done       ", end='\r')
    print("Computing editdistance and writing to -> " + 'results.' + model_name.split('/')[-1])
    codecs.open('results.' + model_name.split('/')[-1] ,'w',encoding='utf-8').write(sentence_in + '\n' + 
                                                               sentence_real.replace(u'\u3233',u'ու').replace(u'\u3234',u'Ու').replace(u'\u3235',u'ՈՒ') + '\n' +
                                                               sentence_out.replace(u'\u3233',u'ու').replace(u'\u3234',u'Ու').replace(u'\u3235',u'ՈՒ') + '\n' +
                                                               str(editdistance.eval(sentence_real,sentence_out)) + ' ' + str(len(sentence_real)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdim', default=512, type=int)
    parser.add_argument('--input', default=None)
    parser.add_argument('--seq_len', default=40, type=int)
    parser.add_argument('--model', default=None)
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--translit', default=None)
    
    args = parser.parse_args()
   
    print("Building network ...")
   
    (output_layer, guess) = define_model(args.hdim, args.depth)
    
    if args.model:
        f = np.load(args.model)
        param_values = [np.float32(f[i]) for i in range(len(f))]
        lasagne.layers.set_all_param_values(output_layer, param_values)
    print("Testing ...")
    
    if args.translit:
        translate_translit(guess, args.translit, args.seq_len)
    else:
        try_it_out(guess, args.input, args.model, args.seq_len)
    
if __name__ == '__main__':
    main()

