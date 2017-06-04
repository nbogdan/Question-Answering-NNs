
lines = [line.strip('\n').split('\t') for line in open('datum.txt', encoding='utf8')]

lines_true_test = list(filter(lambda x: x[3]=="1", lines[1:1201]))
lines_false_test = list(filter(lambda x: x[3]=="0", lines[1:1201]))
lines_true_train = list(filter(lambda x: x[3]=="1", lines[1201:]))
lines_false_train = list(filter(lambda x: x[3]=="0", lines[1201:]))

# lines_true_train = list(filter(lambda x: len(x) == 4 and x[3]=="1", lines))
# lines_false_train = list(filter(lambda x: len(x) == 4 and x[3]=="0", lines))

f2 = open('train_datum_.txt', 'w', encoding='utf8')
f = open('test_datum_.txt', 'w', encoding='utf8')
f3 = open('classic_test_datum_.txt', 'w', encoding='utf8')
f.write('\t'.join(lines[0]) + '\n')
f2.write('\t'.join(lines[0]) + '\n')
f3.write('\t'.join(lines[0]) + '\n')
for i in range(len(lines_true_test)):
    f3.write('\t'.join(lines_true_test[i]) + '\n')
    f3.write('\t'.join(lines_false_test[i * 3]) + '\n')
    f3.write('\t'.join(lines_false_test[i * 3 + 1]) + '\n')
    f3.write('\t'.join(lines_false_test[i * 3 + 2]) + '\n')
for i in range(len(lines_false_test)):
    f.write('\t'.join(lines_true_test[i % len(lines_true_test)]) + '\n')
    f.write('\t'.join(lines_false_test[i]) + '\n')
for i in range(len(lines_false_train)):
    f2.write('\t'.join(lines_true_train[i % len(lines_true_train)]) + '\n')
    f2.write('\t'.join(lines_false_train[i]) + '\n')
f.close()
f2.close()
f3.close()