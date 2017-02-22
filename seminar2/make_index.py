#!/usr/bin/env python
import os
import cPickle
import argparse
from collections import defaultdict

from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize


def build_index(dir_path):
    index = dict()
    id_to_path = dict()
    document_id = 0
    for document_name in os.listdir(dir_path):
        if document_name.endswith(".txt"):
            document_path = os.path.join(dir_path, document_name)
            id_to_path[document_id] = document_path
            with open(document_path) as f:
                index[document_id] = f.read()
            document_id += 1
    return index, id_to_path


def build_inverted_index(index):
    inverted_index = defaultdict(lambda: set())
    for document_id, content in index.iteritems():
        for word in tokenize(content):
            inverted_index[word.lower()].add(document_id)
    return dict(inverted_index)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--dir", nargs="?", default="./txt/", help="Path to the directory with documents")

    dir_path = arg_parser.parse_args().dir

    index, id_to_path = build_index(dir_path)
    inverted_index = build_inverted_index(index)
    cPickle.dump((inverted_index, id_to_path), open("iindex.pkl", "wb"))


if __name__ == "__main__":
    main()
