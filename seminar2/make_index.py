#!/usr/bin/env python
import os
import cPickle
import argparse
from collections import defaultdict, Counter

import numpy as np
from numpy import log2

from utils import get_terms, PICKLE_PATH

EPSILON = 1e-6


def build_index(dir_path):
    index = dict()
    id_to_path = dict()
    document_id = 0
    for document_name in os.listdir(dir_path):
        if document_name.endswith(".txt"):
            document_path = os.path.join(dir_path, document_name)
            with open(document_path) as f:
                index[document_id] = f.read()
            id_to_path[document_id] = document_path
            document_id += 1
    return index, id_to_path


def get_vocabulary(index):
    vocabulary = {}
    
    for document_id, document in index.iteritems():
        tokens = get_terms(document)
        if not len(tokens):
            continue
        vocabulary[]


def build_inverted_index(index):
    inverted_index = defaultdict(lambda: dict())
    for document_id, document in index.iteritems():
        tokens = get_terms(document)
        if not len(tokens):
            continue
        terms_frequencies = Counter(tokens)
        max_frequency =  terms_frequencies.most_common(1)[0][1]
        for term, frequency in terms_frequencies.iteritems():
            tf = float(frequency) / max_frequency  # 0 <= tf <= 1
            inverted_index[term][document_id] = tf
    return dict(inverted_index)


def get_documents_number(dir_path):
    return len(filter(lambda name: name.endswith(".txt"), os.listdir(dir_path)))


def get_inverse_document_frequencies(inverted_index, num_docs):
    term_to_idf = dict()
    for term, posting_list in inverted_index.iteritems():
        num_docs_local = len(posting_list)
        idf = log2(float(num_docs) / num_docs_local) / log2(num_docs) # 0 <= idf <= 1
        term_to_idf[term] = idf
    return term_to_idf


def get_npmi_matrix(inverted_index, num_docs):
    vocabulary = inverted_index.keys()
    voc_size = len(inverted_index)
    npmi_matrix = np.array((voc_size, voc_size))
    for v1 in vocabulary:
        p1 = float(inverted_index[v1]) / num_docs
        for v2 in vocabulary:
            p2 = float(inverted_index[v2]) / num_docs
            p12 = float(inverted_index[v1] & inverted_index[v2]) / num_docs + EPSILON
            npmi_matrix[v1]


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--dir", nargs="?", default="./txt/", help="Path to the directory with documents")

    dir_path = arg_parser.parse_args().dir

    index, id_to_path = build_index(dir_path)
    inverted_index = build_inverted_index(index)
    num_docs = get_documents_number(dir_path)
    term_to_idf = get_inverse_document_frequencies(inverted_index, num_docs)
    cPickle.dump((inverted_index, id_to_path, term_to_idf), open(PICKLE_PATH, "wb"))


if __name__ == "__main__":
    main()
