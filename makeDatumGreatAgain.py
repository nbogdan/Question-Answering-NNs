import numpy as np
import pickle
import random
FOLDER = './data_test_extra/data/'

def split_data():
    lines = [line.strip('\n').split('\t') for line in open('datum_extra.txt', encoding='utf8')]
    lines_true_test = list(filter(lambda x: x[3]=="1", lines[1:401] + lines[4001:4401] + lines[9001:9401]))
    lines_false_test = list(filter(lambda x: x[3]=="0", lines[1:401] + lines[4001:4401] + lines[9001:9401]))
    not_test = list(filter(lambda x: x not in lines_true_test + lines_false_test, lines))
    lines_true_train = list(filter(lambda x: x[3]=="1", not_test))
    lines_false_train = list(filter(lambda x: x[3]=="0", not_test))

    # lines_true_train = list(filter(lambda x: len(x) == 4 and x[3]=="1", lines))
    # lines_false_train = list(filter(lambda x: len(x) == 4 and x[3]=="0", lines))

    f2 = open(FOLDER + 'train_datum.txt', 'w', encoding='utf8')
    f = open(FOLDER + 'test_datum.txt', 'w', encoding='utf8')
    f3 = open(FOLDER + 'classic_test_datum.txt', 'w', encoding='utf8')
    f.write('\t'.join(lines[0]) + '\n')
    f2.write('\t'.join(lines[0]) + '\n')
    f3.write('\t'.join(lines[0]) + '\n')
    for i in range(len(lines_true_test)):
        answerList = [lines_true_test[i]] + lines_false_test[i * 3:i * 3 + 3]
        random.shuffle(answerList)
        for x in range(4):
            f3.write('\t'.join(answerList[x]) + '\n')
    for i in range(len(lines_false_test)):
        f.write('\t'.join(lines_true_test[i % len(lines_true_test)]) + '\n')
        f.write('\t'.join(lines_false_test[i]) + '\n')
    for i in range(len(lines_false_train)):
        f2.write('\t'.join(lines_true_train[i % len(lines_true_train)]) + '\n')
        f2.write('\t'.join(lines_false_train[i]) + '\n')
    f.close()
    f2.close()
    f3.close()
split_data()
def eliminate_duplicates():
    FOLDER = './data_test_extra/'
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

# print(lines[0])
# COUNT_0 = [len(x[0].split(" ")) for x in lines[:10000]]
# COUNT_1 = [len(x[1].split(" ")) for x in lines[:10000]]
# COUNT_2 = [len(x[2].split(" ")) for x in lines[:10000]]
# print(sum(COUNT_0) / len(COUNT_0))
# print(max(COUNT_0))
# id = np.argmax(COUNT_0)
# print(lines[id][0])
#
# print(sum(COUNT_1) / len(COUNT_1))
# print(max(COUNT_1))
# id = np.argmax(COUNT_1)
# print(lines[id][1])
# print(sum(COUNT_2) / len(COUNT_2))
# print(max(COUNT_2))
# print(str(len(list(filter(lambda x: x > 500, COUNT_2)))))
# id = np.argmax(COUNT_2)
# print(lines[id][2])

