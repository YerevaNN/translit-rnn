# -*- coding: utf-8 -*
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
import codecs
import json
import random
from lasagne.init import Orthogonal, Normal

#Lasagne Seed for Reproducibility
random.seed(1)
lasagne.random.set_rng(np.random.RandomState(1))

def isNativeLetter(s, transliteration):
    
    ### Checks if a character is from a native languge
    
    for c in s:
        if c not in transliteration:
            return False
    return True

def valid(transliteration, sequence):
    
    ### Replaces all non given characters with '#'
    valids = [u'\u2000', u'\u2001',';',':','-',',',' ','\n','\t'] + \
             [chr(ord('0') + i) for i in range(10)] + \
             list(set(''.join([''.join([s for s in transliteration[c]]) for c in transliteration])))
    
    ans = []
    non_valids = []
    
    for c in sequence:
        if c in valids:
            ans.append(c)
        else:
            ans.append('#')
            non_valids.append(c)
    return (ans,non_valids)

def toTranslit(prevc,c,nextc,trans):    
    
    ### Translates one character to translit, given probabilistic mapping. 
    ### Previous and next characters are given for some armenian-specific parts
    
    if not isNativeLetter(c, trans):
        return c
        
    ### Armenian Specific Snippet
    
    if(c == u'ո'):
        if(isNativeLetter(prevc, trans)):
            return u'o'
        return u'vo'
    if(c == u'Ո'):
        if(isNativeLetter(prevc, trans)):
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


def make_vocabulary_files(data, language, transliteration):

    ### Makes jsons for future mapping of letters to indices and vice versa

    pointer = 0
    done = False
    s_l = 100000
    chars = set()
    trans_chars = set()
    data = ' \t' + u'\u2001'  + data # to get these symbols in vocab
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
            trans_char = toTranslit(raw_native[ind-1], raw_native[ind], raw_native[ind+1], transliteration)
            translit.append(trans_char[0])
            native.append(raw_native[ind])
            if len(trans_char) > 1:
                native.append(u'\u2000')
                translit.append(trans_char[1])

        translit = valid(transliteration, translit)[0]
        for i in range(len(native)):
            if translit[i] == '#':
                native[i] = '#'
        chars = chars.union(set(native))
        trans_chars = trans_chars.union(set(translit))
        print(str(100.0*pointer/len(data)) + "% done       ", end='\r')
        
    chars = list(chars)
    char_to_index = { chars[i] : i for i in range(len(chars)) }
    index_to_char = { i : chars[i] for i in range(len(chars)) }
    
    open('languages/' + language + '/char_to_index.json','w').write(json.dumps(char_to_index))
    open('languages/' + language + '/index_to_char.json','w').write(json.dumps(index_to_char))
    
    trans_chars = list(trans_chars)
    trans_to_index = { trans_chars[i] : i for i in range(len(trans_chars)) }
    index_to_trans = { i : trans_chars[i] for i in range(len(trans_chars)) }
    trans_vocab_size = len(trans_chars)
    
    open('languages/' + language + '/trans_to_index.json','w').write(json.dumps(trans_to_index))
    open('languages/' + language + '/index_to_trans.json','w').write(json.dumps(index_to_trans))

def load_vocabulary(language):
    
    ### Loads vocabulary mappings from specified json files

    char_to_index = json.loads(open('languages/' + language + '/char_to_index.json').read())
    char_to_index = { i : int(char_to_index[i]) for i in char_to_index}
    
    index_to_char = json.loads(open('languages/' + language + '/index_to_char.json').read())
    index_to_char = { int(i) : index_to_char[i] for i in index_to_char}
    vocab_size = len(char_to_index)
    
    
    trans_to_index = json.loads(open('languages/' + language + '/trans_to_index.json').read())
    trans_to_index = { i : int(trans_to_index[i]) for i in trans_to_index}
    
    index_to_trans = json.loads(open('languages/' + language + '/index_to_trans.json').read())
    index_to_trans = { int(i) : index_to_trans[i] for i in index_to_trans}
    trans_vocab_size = len(trans_to_index)
    return (char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size)

def one_hot_matrix_to_sentence(data, index_to_character):

    ### Converts one sequence of one hot vectors to a string sentence

    if data.shape[0] == 1:
        data = data[0]
    sentence = ""
    for i in data:
        sentence += index_to_character[np.argmax(i)]
    return sentence

def load_language_data(language, is_train = True):
    
    TEST_DATA_PATH = 'languages/' + language + '/data/test.txt'
    VALIDATION_DATA_PATH = 'languages/' + language + '/data/val.txt'
    TRAIN_DATA_PATH = 'languages/' + language + '/data/train.txt'
    long_letters = json.loads(codecs.open('languages/' + language + '/long_letters.json','r',encoding='utf-8').read())
    long_letter_mapping = { long_letters[i] : unichr(ord(u'\u2002') + i) for i in range(len(long_letters)) }
    trans = json.loads(codecs.open('languages/' + language + '/transliteration.json','r',encoding='utf-8').read())
    tmp_trans = trans.copy()
    for c in tmp_trans:
        if c in long_letters:
            trans[long_letter_mapping[c]] = trans[c]
    del tmp_trans
    
    if is_train:
        train_text = codecs.open(TRAIN_DATA_PATH, encoding='utf-8').read()
        val_text = codecs.open(VALIDATION_DATA_PATH, encoding='utf-8').read()
        
        for letter in long_letter_mapping:
            train_text = train_text.replace(letter,long_letter_mapping[letter])
            val_text = val_text.replace(letter,long_letter_mapping[letter])
        
        return (train_text, val_text, trans)
    else:
        test_text = codecs.open(TEST_DATA_PATH, encoding='utf-8').read()
        for letter in long_letter_mapping:
            test_text = test_text.replace(letter,long_letter_mapping[letter])
        long_letter_reverse_mapping = { long_letter_mapping[i] : i for i in long_letter_mapping } 
        
        return (test_text, trans, long_letter_reverse_mapping)
        
def gen_data(p, seq_len, batch_size, data, transliteration, trans_to_index, char_to_index, is_train = True):
    
    ### Generates training examples from data, starting from given index
    ### and returns the index where it stopped
    ### also returns the number of sequences skipped (because of lack of native characters)
    ### and a boolean showing whether generation passed one iteration over data or not
        
    trans_vocab_size = len(trans_to_index)
    vocab_size = len(char_to_index)
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
                    raw_native = ' ' + raw_native[:new_p+1] + ' '
                    p += new_p + 1
                else:
                    p = new_p + 1
                    raw_native = ' ' + raw_native + ' '
            else:
                raw_native = ' ' + raw_native + ' '
                p = 0
                turned = True
            native_letter_count = sum([1 for c in raw_native if isNativeLetter(c, transliteration)])
            if not is_train or native_letter_count * 3 > len(raw_native):
                break
            else:
                non_native_sequences += 1

        native = []
        translit = []
        for ind in range(1,len(raw_native)-1):
            trans_char = toTranslit(raw_native[ind-1], raw_native[ind], raw_native[ind+1], transliteration)
            translit.append(trans_char[0])
            trans_ind = 1
            native.append(raw_native[ind])
            while len(trans_char) > trans_ind:
                native.append(u'\u2000')
                translit.append(trans_char[trans_ind])
                trans_ind += 1
            
        (translit,non_valids) = valid(transliteration, translit)
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
    
    if is_train:
        return (x,y,p,turned,non_native_sequences)
    
    else:
        return (x,y,non_valids,p,turned)

def define_model(N_HIDDEN, depth, LEARNING_RATE = 0.01,  GRAD_CLIP = 100, trans_vocab_size=0, vocab_size=0, is_train = False):
    
    ### Defines lasagne model
    ### Returns output layer and theano functions for training and computing the cost
    
    l_input = lasagne.layers.InputLayer(shape=(None, None, trans_vocab_size))
    network = l_input
    symbolic_batch_size = lasagne.layers.get_output(network).shape[0]
    
    while depth > 0 :
        
        l_forward = lasagne.layers.LSTMLayer(
            network, N_HIDDEN, grad_clipping=GRAD_CLIP,
            ingate=lasagne.layers.Gate(
                                    W_in=Orthogonal(gain=1.5),
                                    W_hid=Orthogonal(gain=1.5),
                                    W_cell=Normal(0.1)),
            forgetgate=lasagne.layers.Gate(
                                    W_in=Orthogonal(gain=1.5),
                                    W_hid=Orthogonal(gain=1.5),
                                    W_cell=Normal(0.1)),
            cell=lasagne.layers.Gate(W_cell=None,
                                    nonlinearity=lasagne.nonlinearities.tanh,
                                    W_in=Orthogonal(gain=1.5),
                                    W_hid=Orthogonal(gain=1.5)),
            outgate=lasagne.layers.Gate(
                                    W_in=Orthogonal(gain=1.5),
                                    W_hid=Orthogonal(gain=1.5),
                                    W_cell=Normal(0.1)),
            backwards=False
            )
        l_backward = lasagne.layers.LSTMLayer(
            network, N_HIDDEN, grad_clipping=GRAD_CLIP,
            ingate=lasagne.layers.Gate(
                                    W_in=Orthogonal(gain=1.5),
                                    W_hid=Orthogonal(gain=1.5),
                                    W_cell=Normal(0.1)),
            forgetgate=lasagne.layers.Gate(
                                    W_in=Orthogonal(gain=1.5),
                                    W_hid=Orthogonal(gain=1.5),
                                    W_cell=Normal(0.1)),
            cell=lasagne.layers.Gate(W_cell=None,
                                    nonlinearity=lasagne.nonlinearities.tanh,
                                    W_in=Orthogonal(gain=1.5),
                                    W_hid=Orthogonal(gain=1.5)),
            outgate=lasagne.layers.Gate(
                                    W_in=Orthogonal(gain=1.5),
                                    W_hid=Orthogonal(gain=1.5),
                                    W_cell=Normal(0.1)),
            backwards=True
            )
        
        if depth == 1:
            l_cell_forward = LSTMLayer(
                network, N_HIDDEN, grad_clipping=GRAD_CLIP,
                ingate=lasagne.layers.Gate(
                                        W_in=l_forward.W_in_to_ingate,
                                        W_hid=l_forward.W_hid_to_ingate,
#                                        W_cell=l_forward.W_cell_to_ingate,
                                        b=l_forward.b_ingate),
                forgetgate=lasagne.layers.Gate(
                                        W_in=l_forward.W_in_to_forgetgate,
                                        W_hid=l_forward.W_hid_to_forgetgate,
#                                       W_cell=l_forward.W_cell_to_forgetgate,
                                        b=l_forward.b_forgetgate),
                cell=lasagne.layers.Gate(W_cell=None,
                                        nonlinearity=lasagne.nonlinearities.tanh,
                                        W_in=l_forward.W_in_to_cell,
                                        W_hid=l_forward.W_hid_to_cell,
                                        b=l_forward.b_cell),
                outgate=lasagne.layers.Gate(
                                        W_in=l_forward.W_in_to_outgate,
                                        W_hid=l_forward.W_hid_to_outgate,
#                                        W_cell=l_forward.W_cell_to_outgate,
                                        b=l_forward.b_outgate),
                backwards=False,
                peepholes=False)
            
            l_cell_backwards = LSTMLayer(
                network, N_HIDDEN, grad_clipping=GRAD_CLIP,
                ingate=lasagne.layers.Gate(
                                        W_in=l_backward.W_in_to_ingate,
                                        W_hid=l_backward.W_hid_to_ingate,
#                                        W_cell=l_backward.W_cell_to_ingate,
                                        b=l_backward.b_ingate),
                forgetgate=lasagne.layers.Gate(
                                        W_in=l_backward.W_in_to_forgetgate,
                                        W_hid=l_backward.W_hid_to_forgetgate,
#                                        W_cell=l_backward.W_cell_to_forgetgate,
                                        b=l_backward.b_forgetgate),
                cell=lasagne.layers.Gate(W_cell=None,
                                        nonlinearity=lasagne.nonlinearities.tanh,
                                        W_in=l_backward.W_in_to_cell,
                                        W_hid=l_backward.W_hid_to_cell,
                                        b=l_backward.b_cell),
                outgate=lasagne.layers.Gate(
                                        W_in=l_backward.W_in_to_outgate,
                                        W_hid=l_backward.W_hid_to_outgate,
#                                        W_cell=l_backward.W_cell_to_outgate,
                                        b=l_backward.b_outgate),
                backwards=True,
                peepholes=False)
        
        concat_layer = lasagne.layers.ConcatLayer(incomings=[l_forward, l_backward], axis = 2)
        concat_layer = lasagne.layers.ReshapeLayer(concat_layer, (-1, 2*N_HIDDEN))
        network = lasagne.layers.DenseLayer(concat_layer, num_units=N_HIDDEN, W = Orthogonal(), nonlinearity=lasagne.nonlinearities.tanh)
        network = lasagne.layers.ReshapeLayer(network, (symbolic_batch_size, -1, N_HIDDEN))
        
        depth -= 1
    
    
    
    network = lasagne.layers.ReshapeLayer(network, (-1, N_HIDDEN) )
    l_input_reshape = lasagne.layers.ReshapeLayer(l_input, (-1, trans_vocab_size))
    network = lasagne.layers.ConcatLayer(incomings=[network,l_input_reshape], axis = 1)
    
    l_out = lasagne.layers.DenseLayer(network, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.dmatrix('target_output')
    
    network_output = lasagne.layers.get_output(l_out)
    network = lasagne.layers.get_output(network)
    concat_layer = lasagne.layers.get_output(concat_layer)
    last_lstm_cells_forward = lasagne.layers.get_output(l_cell_forward)
    last_lstm_cells_backwards = lasagne.layers.get_output(l_cell_backwards)
    #gates = l_cell_forward.get_gates()


    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    
    print("Compiling Functions ...")
    
    if is_train:
        
        print("Computing Updates ...")
        #updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
        updates = lasagne.updates.adam(cost, all_params, beta1=0.5, learning_rate=LEARNING_RATE) # from DCGAN paper
        
        compute_cost = theano.function([l_input.input_var, target_values], cost, allow_input_downcast=True)
        train = theano.function([l_input.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
        return(l_out, train, compute_cost)
    
    else:
        guess = theano.function([l_input.input_var],
                                [network_output, network, concat_layer,
                                last_lstm_cells_forward, last_lstm_cells_backwards
                                #gates[0], gates[1], gates[2]
                                ],
                                allow_input_downcast=True)
        return(l_out, guess)

def isDelimiter(c):
    return c in [u'\n', u'\t', u' ']

def chunk_parse(chunk, seq_len, batch_size, transliteration, trans_to_index, char_to_index, is_train = False):
        
    trans_vocab_size = len(trans_to_index)
    vocab_size = len(char_to_index)
    
    delimiters = [u'']
    words = []
    word = ''
    i = 0
    while i < len(chunk):
        if isDelimiter(chunk[i]):
            words.append(word)
            word = ''
            delimiter = chunk[i]
            while i+1 < len(chunk) and isDelimiter(chunk[i+1]):
                i += 1
                delimiter += chunk[i]
            delimiters.append(delimiter)
        else:
            word += chunk[i]
            
        i += 1
        
    if word != '':
        words.append(word)
    
    sequences = []
    s = ""
    sequence_delimiters = [u'']
    
    for (word,delimiter) in zip(words,delimiters):
        if len(s) + len(word) <= seq_len:
            s += delimiter + word
        elif len(s) != 0:
            sequences.append(s);
            s = word
            sequence_delimiters.append(delimiter)
    
    if s != '':
        sequences.append(s)
    
    samples = []
    for seq in sequences:
        
        native_letter_count = sum([1 for c in seq if isNativeLetter(c, transliteration)])
        if is_train and native_letter_count * 3 < len(seq):
            continue
        
        seq = u' ' + seq + u' '
        translit = []
        native = []
        for ind in range(1,len(seq)-1):
            trans_char = toTranslit(seq[ind-1], seq[ind], seq[ind+1], transliteration)
            translit.append(trans_char[0])
            trans_ind = 1          
            native.append(seq[ind])
            while len(trans_char) > trans_ind:
                native.append(u'\u2000')
                translit.append(trans_char[trans_ind])
                trans_ind += 1      

        translit, non_valids = valid(transliteration, translit)
        for ind in range(len(native)):
            if translit[ind] == '#':
                native[ind] = '#' 
        samples.append( (translit, native, non_valids))

        '''
        translits.append(translit)
        natives.append(native)
        non_valids_list.append(non_valids)
        '''
    if is_train:
        samples.sort(key = lambda x: len(x[0]), reverse = True)
        
        buckets = {}
        for tmp in samples:
            if len(tmp[0]) not in buckets.keys():
                buckets[len(tmp[0])] = []
            buckets[len(tmp[0])].append(tmp)
            
        del samples
        for i in buckets:
            random.shuffle(buckets[i])
        
        batches = []
        for i in buckets.keys():
            j = 0
            while j < len(buckets[i]):
                batches.append(list(buckets[i][j:j+batch_size]))
                j += batch_size
        del buckets
        
        np_batches = []
        
        for batch in batches:
            x = np.zeros( (len(batch), len(batch[0][0]), trans_vocab_size) )
            y = np.zeros( (len(batch), len(batch[0][0]), vocab_size) )
            for i in range(len(batch)):
                for j in range(len(batch[i][0])):
                    x[i, j, trans_to_index[batch[i][0][j]] ] = 1
                    y[i, j, char_to_index[batch[i][1][j]] ] = 1
            np_batches.append((x,y))
        
        return np_batches
    
    else:
        indexed_samples = sorted(zip(samples, range(len(samples)), sequence_delimiters) , key = lambda x: (len(x[0][0]), x[1]) , reverse = True)
        #samples, indices = zip(indexed_samples)
        #del indexed_samples
        
        buckets = {}
        for tmp in indexed_samples:
            if len(tmp[0][0]) not in buckets.keys():
                buckets[len(tmp[0][0])] = []
            buckets[len(tmp[0][0])].append(tmp)
            
        del indexed_samples
        
        batches = []
        for i in buckets.keys():
            j = 0
            while j < len(buckets[i]):
                batches.append(list(buckets[i][j:j+batch_size]))
                j += batch_size
        del buckets
        
        np_batches = []
        non_vals = []
        
        
        for batch in batches:
            indices = np.zeros( len(batch) )
            delimiters = [0] * len(batch)
            x = np.zeros( (len(batch), len(batch[0][0][0]), trans_vocab_size) )
            y = np.zeros( (len(batch), len(batch[0][0][0]), vocab_size) )
            non_vals.append([])
            for i in range(len(batch)):
                indices[i] = batch[i][1]
                delimiters[i] = batch[i][2]
                non_vals[-1].append(batch[i][0][2])
                for j in range(len(batch[i][0][0])):
                    x[i, j, trans_to_index[batch[i][0][0][j]] ] = 1
                    y[i, j, char_to_index[batch[i][0][1][j]] ] = 1
            np_batches.append( (x, y, indices, delimiters) )
            
        return (np_batches, non_vals)

def data_generator(data, seq_len, batch_size, transliteration, trans_to_index, char_to_index, is_train = False):
    p = 0
    while p < len(data):
        if is_train:
            parsed_data = chunk_parse(data[p:p+700000], seq_len, batch_size, transliteration, trans_to_index, char_to_index, is_train)
            p += 700000
            random.shuffle(parsed_data)
            for batch in parsed_data:
                yield batch
        else:
            parsed_data, non_valids = chunk_parse(data[p:p+700000], seq_len, batch_size, transliteration, trans_to_index, char_to_index, is_train)
            p += 700000
            for batch in zip(parsed_data, non_valids):
                yield batch
"""
def temp_data_generator(data, seq_len, batch_size, transliteration, trans_to_index, char_to_index, is_train = False):
    p = 0
    size = 7000000 / 35
    while p < len(data):
        if is_train:
            p += 7000000
            parsed_data = []
            for i in range(35):
                start_index = np.random().randint(7000000 - size + 1)

                parsed_data.extend(chunk_parse(data[start_index:start_index + size], seq_len, batch_size, transliteration, trans_to_index, char_to_index, is_train))
            random.shuffle(parsed_data)
            for batch in parsed_data:
                yield batch
        else:
            parsed_data, non_valids = chunk_parse(data[p:p+7000000], seq_len, batch_size, transliteration, trans_to_index, char_to_index, is_train)
            p += 7000000
            for batch in zip(parsed_data, non_valids):
                yield batch
"""

from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper

__all__ = [
    "CustomRecurrentLayer",
    "RecurrentLayer",
    "Gate",
    "LSTMLayer",
    "GRULayer"
]


class Gate(object):
    """
    lasagne.layers.recurrent.Gate(W_in=lasagne.init.Normal(0.1),
    W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)
    Simple class to hold the parameters for a gate connection.  We define
    a gate loosely as something which computes the linear mix of two inputs,
    optionally computes an element-wise product with a third, adds a bias, and
    applies a nonlinearity.
    Parameters
    ----------
    W_in : Theano shared variable, numpy array or callable
        Initializer for input-to-gate weight matrix.
    W_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-gate weight matrix.
    W_cell : Theano shared variable, numpy array, callable, or None
        Initializer for cell-to-gate weight vector.  If None, no cell-to-gate
        weight vector will be stored.
    b : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector.
    nonlinearity : callable or None
        The nonlinearity that is applied to the input gate activation. If None
        is provided, no nonlinearity will be applied.
    Examples
    --------
    For :class:`LSTMLayer` the bias of the forget gate is often initialized to
    a large positive value to encourage the layer initially remember the cell
    value, see e.g. [1]_ page 15.
    >>> import lasagne
    >>> forget_gate = Gate(b=lasagne.init.Constant(5.0))
    >>> l_lstm = LSTMLayer((10, 20, 30), num_units=10,
    ...                    forgetgate=forget_gate)
    References
    ----------
    .. [1] Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to
           forget: Continual prediction with LSTM." Neural computation 12.10
           (2000): 2451-2471.
    """
    def __init__(self, W_in=init.Normal(0.1), W_hid=init.Normal(0.1),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

class LSTMLayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    A long short-term memory (LSTM) layer.
    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by
    .. math ::
        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(LSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [ingate, forgetgate, outgate, cell_input, cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            ingate, forgetgate, outgate, cell_input, cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[None, None, None, None, cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
            ingate = ingate[-1]
            forgetgate = forgetgate[-1]
            outgate = outgate[-1]
            cell_input = cell_input[-1]
            cell_out = cell_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            ingate = ingate.dimshuffle(1, 0, 2)
            forgetgate = forgetgate.dimshuffle(1, 0, 2)
            outgate = outgate.dimshuffle(1, 0, 2)
            cell_input = cell_input.dimshuffle(1, 0, 2)
            cell_out = cell_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
                ingate = ingate[:, ::-1]
                forgetgate = forgetgate[:, ::-1]
                outgate = outgate[:, ::-1]
                cell_input = cell_input[:, ::-1]
                cell_out = cell_out[:, ::-1]
    
        concat = T.concatenate([ingate, forgetgate, outgate, cell_input, cell_out], axis=2)
        return concat
