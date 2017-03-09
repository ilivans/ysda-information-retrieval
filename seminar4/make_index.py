#!/usr/bin/env python
from __future__ import print_function

import os
import cPickle
import argparse
from collections import Counter
from time import clock
import warnings

import numpy as np
import sys

from numpy import log2
from joblib import Parallel, delayed

from utils import get_terms, make_npmi_dir, PICKLE_PATH, TFIDF_PATH, NPMI_PART_SIZE, NPMI_PATH_TEMPLATE, VOCABULARY_PATH

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
