#!/usr/bin/env python
import os
import cPickle
import argparse
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from numpy import log2
from scipy.special import expit

from utils import get_terms

EPSILON = 1e-3


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


def build_inverted_index(index):
    inverted_index = defaultdict(lambda: list())
    for document_id, document in index.iteritems():
        tokens = get_terms(document)
        terms_frequencies = Counter(tokens)
        for term, frequency in terms_frequencies.iteritems():
            tf = float(frequency) / (len(tokens) + EPSILON)  # 0 <= tf < 1
            inverted_index[term].append([document_id, tf])

    # Inverted index with DataFrame as a value
    inverted_index_df = dict()
    for term, doc_ids_and_frequencies in inverted_index.iteritems():
        inverted_index_df[term] = pd.DataFrame(doc_ids_and_frequencies, columns=["doc_id", term])

    return inverted_index_df


def get_documents_number(dir_path):
    return len(filter(lambda name: name.endswith(".txt"), os.listdir(dir_path)))


def get_inverse_document_frequencies(inverted_index, num_docs):
    term_to_idf = dict()
    for term, doc_ids_and_frequencies in inverted_index.iteritems():
        num_docs_local = doc_ids_and_frequencies.shape[0]
        idf = 2 * expit(log2(max(float(num_docs) / (num_docs_local + EPSILON), 1.))) - 1  # 0 <= idf < 1
        term_to_idf[term] = idf


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--dir", nargs="?", default="./txt/", help="Path to the directory with documents")

    dir_path = arg_parser.parse_args().dir

    index, id_to_path = build_index(dir_path)
    inverted_index = build_inverted_index(index)
    num_docs = get_documents_number(dir_path)
    term_to_idf = get_inverse_document_frequencies(inverted_index, num_docs)
    cPickle.dump((inverted_index, id_to_path, term_to_idf), open("iindex.pkl", "wb"))


if __name__ == "__main__":
    main()
