import numpy as np
import pickle
import random
FOLDER = './data_small/data'


def split_data():
    lines = [line.strip('\n').split('\t') for line in open(FOLDER + 'datum.txt', encoding='utf8')]

    test_lines = lines[1:401]
    lines_true_test = list(filter(lambda x: x[3]=="1", test_lines))
    lines_false_test = list(filter(lambda x: x[3]=="0", test_lines))

    not_test = lines[401:]
    lines_true_train = list(filter(lambda x: x[3]=="1", not_test))
    lines_false_train = list(filter(lambda x: x[3]=="0", not_test))

    f_train = open(FOLDER + 'test_datum.txt', 'w', encoding='utf8')
    f_test = open(FOLDER + 'train_datum.txt', 'w', encoding='utf8')

    f_train.write('\t'.join(lines[0]) + '\n')
    f_test.write('\t'.join(lines[0]) + '\n')

    for i in range(len(lines_true_test)):
        answerList = [lines_true_test[i]] + lines_false_test[i * 3:i * 3 + 3]
        random.shuffle(answerList)
        for x in range(4):
            f_test.write('\t'.join(answerList[x]) + '\n')

    for i in range(len(lines_false_train)):
        f_train.write('\t'.join(lines_true_train[i % len(lines_true_train)]) + '\n')
        f_train.write('\t'.join(lines_false_train[i]) + '\n')
    f_train.close()
    f_test.close()

#not used anymore
def eliminate_duplicates():
    FOLDER = './data_extra/'
    texts_c3 = pickle.load(open(FOLDER + 'train_lemmas_c', 'rb'))
    texts_q3 = pickle.load(open(FOLDER + 'train_lemmas_q', 'rb'))
    texts_a3 = pickle.load(open(FOLDER + 'train_lemmas_a', 'rb'))

    clean_c = []
    clean_q = []
    clean_a = []
    aux = []
    for j in range(len(texts_c3)):
        aux.append(texts_c3[j] + texts_q3[j] + texts_a3[j])
    aux = set(aux)
    for i in range(len(texts_c3)):
        if (texts_c3[i] + texts_q3[i] + texts_a3[i]) in aux:
            clean_c.append(texts_c3[i])
            clean_q.append(texts_q3[i])
            clean_a.append(texts_a3[i])
            aux.remove(texts_c3[i] + texts_q3[i] + texts_a3[i])
            print(i)
        else:
            print(0)
    with open(FOLDER + 'train_lemmas_c2', 'wb') as fp:
        pickle.dump(clean_c, fp)
    with open(FOLDER + 'train_lemmas_q2', 'wb') as fp:
        pickle.dump(clean_q, fp)
    with open(FOLDER + 'train_lemmas_a2', 'wb') as fp:
        pickle.dump(clean_a, fp)

if __name__ == '__main__':
    split_data()