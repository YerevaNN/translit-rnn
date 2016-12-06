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
    parser.add_argument('--language', default='hy-AM')
    args = parser.parse_args()
   
    print("Loading Files")
    (train_text, val_text, trans) = utils.load_language_data(language = args.language)

    print("Making Vocabulary Files")
    utils.make_vocabulary_files(train_text, args.language, trans)
    
if __name__ == '__main__':
    main()
