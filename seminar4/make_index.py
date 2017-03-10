#!/usr/bin/env python
from __future__ import print_function
import os
from collections import Counter

import numpy as np

from utils import get_terms

EPSILON = 1e-8


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
    vocabulary = []
    term_to_id = {}
    term_id = 0
    for document_id, document in index.iteritems():
        tokens = get_terms(document)
        if not len(tokens):
            continue
        for term in tokens:
            if term not in term_to_id:
                term_to_id[term] = term_id
                vocabulary.append(term)
                term_id += 1
    return np.array(vocabulary), term_to_id


def build_inverted_index(index, term_to_id):
    voc_size = len(term_to_id)
    inverted_index = [dict() for _ in xrange(voc_size)]
    for document_id, document in index.iteritems():
        tokens = get_terms(document)
        if not len(tokens):
            continue
        terms_frequencies = Counter(tokens)
        for term, frequency in terms_frequencies.iteritems():
            tf = float(frequency)
            inverted_index[term_to_id[term]][document_id] = tf
    return inverted_index


def get_tf_matrix(inverted_index, num_docs):
    voc_size = len(inverted_index)
    tf_matrix = np.zeros((voc_size, num_docs), dtype=np.float32)
    for term, posting_list in enumerate(inverted_index):
        for doc_id, tf in posting_list.iteritems():
            tf_matrix[term, doc_id] = tf
    return tf_matrix


def normalize(tf_matrix):
    num_docs = tf_matrix.shape[1]
    tf_normalized = tf_matrix / tf_matrix.sum(axis=1).reshape((-1, 1))
    tf_normalized = (tf_normalized * np.ma.log(tf_normalized).filled(0.) / np.log(num_docs)).sum(axis=1).reshape((-1, 1))
    tf_normalized = tf_normalized * np.log2(tf_matrix + 1.)
    return tf_normalized
