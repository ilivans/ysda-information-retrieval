import os
import sys
import cPickle
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer('\w+').tokenize


def build_index(dir_path):
    index = dict()
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".txt"):
            file_path = dir_path + file_name
            index[file_path] = open(file_path, 'r').read()
    return index


def build_inverted_index(index):
    inverted_index = defaultdict(lambda: set())
    for f, text in index.iteritems():
        for w in tokenize(text):
            inverted_index[w.lower()].add(f)
    return inverted_index


if __name__=='__main__':
    if len(sys.argv) == 2:
        dir_path = sys.argv[1]
        if dir_path[-1] != '/':
            dir_path += '/'
    else:
        dir_path = './txt/'

    inverted_index = build_inverted_index(build_index(dir_path))
    cPickle.dump(dict(inverted_index), open('iindex.pkl', 'wb'))
