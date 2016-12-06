# -*- coding: utf-8 -*

from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import lasagne
import codecs
import editdistance
import argparse
import utils

def gen_validation_data(p, data, seq_len, transliteration, trans_vocab_size, trans_to_index):
    
    x = np.zeros((1,int(seq_len),trans_vocab_size))
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
    (translit,non_valids) = utils.valid(raw_translit, transliteration)
    for ind in range(len(translit)):
        x[0,ind,trans_to_index[translit[ind]]] = 1
    for ind in range(len(translit),int(seq_len)):
        x[0,ind,trans_to_index[u'\u2001']] = 1
    
    return (x,non_valids,p,turned)
         

def get_residual_weight_matrix(network,csv_name, index_to_char, index_to_trans):
    W = network.get_params()[0].get_value()
    f = open('matrix.csv','w')
    for i in range(len(W)):
        f.write(','.join([str(x) for x in W[i]]) + '\n')
    return
    print(W.shape)
    W = W[-len(index_to_trans):,:]
    print(W.shape)
    fr = ['" "'] + ['"' + index_to_char[i] + '"' for i in range(len(index_to_char))] #152
    rows = [[index_to_trans[i]] + [x for x in W[i] ] for i in range(len(index_to_trans))] #72
    # print(rows)
    codecs.open(csv_name,'w',encoding='utf-8').write(','.join(fr) + '\n' + '\n'.join(['"' + row[0] + '",' + ','.join([ "%.3f" %(r) for r in row[1:] ]) for row in rows]))

def translate_romanized(predict, data, seq_len, transliteration, trans_vocab_size, trans_to_index, index_to_char, long_letter_reverse_mapping):
    p = 0
    turned = False
    sentence_out = "\n"
    while not turned:
        x, non_valids, p, turned = gen_validation_data(p, data, seq_len, transliteration, trans_vocab_size, trans_to_index)
        guess = utils.one_hot_matrix_to_sentence(predict(x),index_to_char).replace(u'\u2001','').replace(u'\u2000','')
        for letter in long_letter_reverse_mapping:
            guess = guess.replace(letter,long_letter_reverse_mapping[letter])

        final_guess = ""
        ind = 0
        for c in guess:
            if c == '#' and ind < len(non_valids):
                final_guess += non_valids[ind]
                ind += 1
            else:
                final_guess += c
        sentence_out += final_guess
        print(str(100.0*p/len(data)) + "% done              ", end='\r')
    print(sentence_out)

def test(predict, data, language, model_name, seq_len, long_letter_reverse_mapping, transliteration, trans_to_index, char_to_index, index_to_trans, index_to_char):
    sentences = []
    p = 0
    
    for ((x_list, y_list, indices, delimiters), non_valids_list) in utils.data_generator(data, seq_len, 100, transliteration, trans_to_index, char_to_index, is_train = False):
        
        guess_list = predict(x_list)[0].reshape(y_list.shape)
        
        for (x, y, guess, non_valids, index, delimiter) in zip(x_list, y_list, guess_list, non_valids_list, indices, delimiters):
            
            sentence_in = utils.one_hot_matrix_to_sentence(x, index_to_trans).replace(u'\u2001','').replace(u'\u2000','')
            real_without_signs = utils.one_hot_matrix_to_sentence(y,index_to_char).replace(u'\u2001','').replace(u'\u2000','')
            guess = utils.one_hot_matrix_to_sentence(guess, index_to_char).replace(u'\u2001','').replace(u'\u2000','')
            
            ind = 0
            sentence_real = ""
            for c in real_without_signs:
                if c == '#' and ind < len(non_valids):
                    sentence_real += non_valids[ind]
                    ind += 1
                else:
                    sentence_real += c
                    
            
            ind = 0
            final_guess = ""
            for c in guess:
                if c == '#' and ind < len(non_valids):
                    final_guess += non_valids[ind]
                    ind += 1
                else:
                    final_guess += c
            final_guess 
            
            sentences.append((index, sentence_in, sentence_real, final_guess, delimiter))
            
            p += len(x)
       # print(str(100.0*p/len(data)) + "% done       ", end='\r')
    
    sentences.sort()
    
    sentence_in = ''.join([ i[4] + i[1]  for i in sentences])
    sentence_real = ''.join([ i[4] + i[2] for i in sentences])
    sentence_out = ''.join([ i[4] + i[3] for i in sentences])
        
    for letter in long_letter_reverse_mapping:
        sentence_real = sentence_real.replace(letter, long_letter_reverse_mapping[letter])
        sentence_out = sentence_out.replace(letter,long_letter_reverse_mapping[letter])
    print("Computing editdistance and writing to -> " + 'languages/' + language + '/results.' + model_name.split('/')[-1])
    
    fl = codecs.open('languages/' + language + '/results.' + model_name.split('/')[-1],'w',encoding='utf-8')
    fl.write(sentence_in + '\n' + sentence_real + '\n' + sentence_out + '\n')
    fl.write(str(editdistance.eval(sentence_real,sentence_out)) + ' / ' + str(len(sentence_real)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdim', default=512, type=int)
    parser.add_argument('--seq_len', default=40, type=int)
    parser.add_argument('--model', default=None)
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--translit_path', default=None)
    parser.add_argument('--language', default=None)
    
    args = parser.parse_args()

    print("Loading Files")
    (char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size) = utils.load_vocabulary(language = args.language)
    (test_text, trans, long_letter_reverse_mapping) = utils.load_language_data(language = args.language, is_train = False)
    print("Building network ...")
    (output_layer, predict) = utils.define_model(args.hdim, args.depth, trans_vocab_size = trans_vocab_size, vocab_size = vocab_size, is_train = False)
    
    # get_residual_weight_matrix(output_layer, "residual", index_to_char, index_to_trans)
    # exit(0)
    #test_text = test_text[:35000]
    
    if args.model:
        f = np.load(args.model)
        param_values = [np.float32(f[i]) for i in range(len(f))]
        lasagne.layers.set_all_param_values(output_layer, param_values)
    print("Testing ...")
    if args.translit_path:
        data = codecs.open(args.translit_path, 'r', encoding='utf-8').read()
        translate_romanized(predict, data, args.seq_len, trans, trans_vocab_size, trans_to_index, index_to_char, long_letter_reverse_mapping)

    else:
        test(predict, test_text, args.language, args.model, args.seq_len, long_letter_reverse_mapping, trans, trans_to_index, char_to_index, index_to_trans, index_to_char)
if __name__ == '__main__':
    main()

