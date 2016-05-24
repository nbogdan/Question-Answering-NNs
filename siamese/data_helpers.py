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

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def read_train():
    #lines = list(open("../data/paraphrase/msr_paraphrase_train.txt", "r").readlines())
    lines = list(open("../../../MSRParaphraseCorpus/msr_paraphrase_train.txt", "r").readlines())
    return lines[1:]

def read_test():
    #lines = list(open("../data/paraphrase/msr_paraphrase_test.txt", "r").readlines())
    lines = list(open("../../../MSRParaphraseCorpus/msr_paraphrase_test.txt", "r").readlines())
    return lines[1:]

def load_data_and_labels(lines):
    col0 = []
    col1 = []
    col2 = []

    lemmatizer = WordNetLemmatizer()

    for line in lines:
        tok = line.split("\t")
        correct_class = int(tok[0])
        if(correct_class == 0):
            col0.append([0, 1])
        else:
            col0.append([1, 0])

        sent1 = clean_str(tok[3]).split(" ")
        sent2 = clean_str(tok[4]).split(" ")

        sent1 = get_lemmas(sent1, lemmatizer)
        sent2 = get_lemmas(sent2, lemmatizer)

        col1.append(sent1)
        col2.append(sent2)

    return [col0, col1, col2]


def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = [[vocabulary[word].index if word != "<PAD/>" and word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences]
    return x


def preprocess_data(model, type):
    if type == "train":
        lines = read_train()
    else:
        lines = read_test()
    labels, sentences1, sentences2 = load_data_and_labels(lines)

    sequence_length1 = max(len(x) for x in sentences1)
    sequence_length2 = max(len(x) for x in sentences2)
    max_len = max(sequence_length1, sequence_length2, 40)
    sent1_padded = pad_sentences(sentences1, max_len)
    sent2_padded = pad_sentences(sentences2, max_len)

    new_sent1 = build_input_data(sent1_padded, model.vocab)
    new_sent2 = build_input_data(sent2_padded, model.vocab)
    return [np.array(labels), np.array(new_sent1), np.array(new_sent2)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
