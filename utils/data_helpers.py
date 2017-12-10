import numpy as np
import re
import nltk
import itertools
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_lemmas(sent, lemmatizer):
    stop_words = []
    res = []
    for word in sent:
        pos = get_wordnet_pos(nltk.pos_tag([word])[0][1])
        if pos == '':
            lemma = lemmatizer.lemmatize(word)
        else:
            lemma = lemmatizer.lemmatize(word, pos)
        #if(type(lemma) == unicode):
        #    lemma = lemma.encode('ascii', 'ignore')

        if lemma.isdigit():
            res.append('number')
        else:
            res.append(lemma)
    return res

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
