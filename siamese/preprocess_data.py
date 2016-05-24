import pickle
from array import array
import numpy as np
import random
import gensim
import data_helpers

index = []
dict = []
not_started = True

#model = gensim.models.Word2Vec.load_word2vec_format('../../../../word2vec/GoogleNews-vectors-small-300.bin', binary=True)
#model = gensim.models.Word2Vec.load_word2vec_format('../../../../word2vec/GoogleNews-vectors-paraphrase-300.bin', binary=True)
#model.init_sims(replace=True)

def clean_sent_cond(sent):
    tok = sent.split(" ")
    if(len(tok) < 13):
        return True
    if max(map(lambda x: len(x), tok)) < 4:
        return True
    return False

def preprocess(model):
    dict_sentences = {}
    reverse_dict = {}
    match_dictionary = {}
    pair_list = []
    import sys
    i = 0
    k = 0
    maxlen = 0
    # this reads in one line at a time from stdin
    for line in sys.stdin:
        i+=1
        tokens = line.split("\t")
        sent1 = tokens[0]
        sent2 = tokens[1]

        if clean_sent_cond(sent1) or clean_sent_cond(sent2):
            continue
        else:
            k += 1

        if not sent1 in dict_sentences:
            dict_sentences[sent1] = len(dict_sentences) + 1
        if not sent2 in dict_sentences:
            dict_sentences[sent2] = len(dict_sentences) + 1
        index_1 = dict_sentences[sent1]
        index_2 = dict_sentences[sent2]

        if not index_1 in match_dictionary:
            match_dictionary[index_1] = []
        if not index_2 in match_dictionary:
            match_dictionary[index_2] = []
        match_dictionary[index_1].append(index_2)
        match_dictionary[index_2].append(index_1)
        pair_list.append((index_1, index_2))

        if i % 10000 == 0:
            print(str(k) + "/" + str(i))
        if k == 500000:
            break;

    i = 0
    for entry in dict_sentences:
        simple_sent1 = filter(lambda x: len(x) > 1, data_helpers.clean_str(entry).split(" "))
        sent1 = data_helpers.build_input_data(data_helpers.pad_sentences([simple_sent1], 40, padding_word="<PAD/>"),
                                          model.vocab)
        reverse_dict[dict_sentences[entry]] = sent1
        if i % 10000 == 0:
            print(i)
        i += 1

    random.shuffle(pair_list)
    pickle.dump(reverse_dict, open("sentences_small_x", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("writing sentences " + str(len(reverse_dict)))
    pickle.dump(match_dictionary, open("pairs_index_small_x", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("writing map " + str(len(match_dictionary)))
    pickle.dump(pair_list, open("pairs_list_small_x", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("pairs " + str(len(pair_list)))


def read_and_index():
    dict = pickle.load(open("sentences_small_x", "rb"))
    print("opened dict")
    index = pickle.load(open("pairs_index_small_x", "rb"))
    print("opened index")
    pairs_list = pickle.load(open("pairs_list_small_x", "rb"))
    print("opened index")

    random.seed(0)
    return index, dict, pairs_list

def print_prop(numbers, model):
    stringA = ""
    for x in numbers:
        if x < len(model.index2word):
            stringA += model.index2word[x].encode('ascii', 'ignore') + " "
    return stringA

def generate_train_batch(pos, batchsize, index, dict, pairs_list):
    batch = []
    for i in range(int(batchsize / 2)):
        index_pair = (pos * int(batchsize / 2) + i) % len(index) + 1#random.randint(1, len(index))

        index_1, index_2 = pairs_list[index_pair]
        index_neg = -1
        while index_neg == -1:
            index_neg = random.randint(1, len(index))
            if index_neg in index[index_1]:
                index_neg = -1
        if len(dict[index_neg][0]) != 40 or len(dict[index_1][0]) != 40 or len(dict[index_2][0]) != 40:
            print("Ouch")
            continue

        if len(batch) == 0:
            batch = np.array([dict[index_1], dict[index_2], [1, 0]])
        else:
            batch = np.insert(batch, 0, np.array([dict[index_1], dict[index_2], [1, 0]]))

        batch = np.insert(batch, 0, np.array([dict[index_1], dict[index_neg], [0, 1]]))
    result1 = []
    result2 = []
    result3 = []
    for i in range(int(len(batch) / 3)):
        result1.append(batch[3 * i][0])
        result2.append(batch[3 * i + 1][0])
        result3.append(batch[3 * i + 2])

    return [tuple(result1), tuple(result2), tuple(result3)]

#preprocess(model)

#dict, index, pairs = read_and_index()

#for i in range(1,20):
#    print str(pairs[i][0]) + ":" + str(pairs[i][1]) + "  " + print_prop(index[pairs[i][0]][0], model)
#    print print_prop(index[pairs[i][1]][0], model)
#    print "\n"