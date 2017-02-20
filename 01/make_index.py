#!/usr/bin/env python
import os
import cPickle
import argparse
from collections import defaultdict

from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize


def build_index(dir_path):
    index = dict()
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(dir_path, file_name)
            index[file_path] = open(file_path, "r").read()
    return index


def build_inverted_index(index):
    inverted_index = defaultdict(lambda: set())
    for file_path, text in index.iteritems():
        for word in tokenize(text):
            inverted_index[word.lower()].add(file_path)
    return dict(inverted_index)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--dir", nargs="?", default="./txt/", help="Path to the directory with documents")

    dir_path = arg_parser.parse_args().dir

    inverted_index = build_inverted_index(build_index(dir_path))
    cPickle.dump(inverted_index, open("iindex.pkl", "wb"))


if __name__ == "__main__":
    main()
