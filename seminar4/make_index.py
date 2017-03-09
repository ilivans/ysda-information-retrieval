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


def build_index():
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


def get_vocabulary():
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


def build_inverted_index():
    inverted_index = [dict() for _ in xrange(voc_size)]
    for document_id, document in index.iteritems():
        tokens = get_terms(document)
        if not len(tokens):
            continue
        terms_frequencies = Counter(tokens)
        max_frequency =  terms_frequencies.most_common(1)[0][1]
        for term, frequency in terms_frequencies.iteritems():
            tf = float(frequency) / max_frequency  # 0 <= tf <= 1
            inverted_index[term_to_id[term]][document_id] = tf
    return inverted_index


def get_documents_number():
    return len(filter(lambda name: name.endswith(".txt"), os.listdir(dir_path)))


def get_inverse_document_frequencies():
    idfs = [0] * voc_size
    for term, posting_list in enumerate(inverted_index):
        num_docs_local = len(posting_list)
        idf = log2(float(num_docs) / num_docs_local) / log2(num_docs) # 0 <= idf <= 1
        idfs[term] = idf
    return idfs


def get_tfidf_matrix():
    tfidf_matrix = np.zeros((num_docs, voc_size), dtype=np.float32)
    for term, posting_list in enumerate(inverted_index):
        idf = idfs[term]
        for doc_id, tf in posting_list.iteritems():
            tfidf_matrix[doc_id, term] = tf * idf
    return tfidf_matrix


def build_and_save_npmi_submatrix(terms_batch):
    npmi_submatrix = np.zeros((len(terms_batch), voc_size), dtype=np.float32)
    term0 = terms_batch[0]  # id of the first term in the batch
    for term1 in terms_batch:
        posting_list1 = inverted_index[term1]
        docs1 = set().union(posting_list1)
        p1 = float(len(docs1)) / num_docs
        for term2, posting_list2 in enumerate(inverted_index):
            p2 = float(len(posting_list2)) / num_docs
            intersection_size = len(docs1.intersection(posting_list2))
            p12 = float(intersection_size) / num_docs
            npmi = log2(p12 / p1 / p2) / (-log2(p12)) if p12 != 0. else -1.
            npmi_submatrix[term1 - term0, term2] = npmi
    np.save(NPMI_PATH_TEMPLATE.format(term0, term0 + NPMI_PART_SIZE - 1), npmi_submatrix)


def build_and_save_npmi_matrix():
    print("Building and saving NPMI matrix...")
    num_tasks = voc_size / NPMI_PART_SIZE + min(1, voc_size % NPMI_PART_SIZE)
    print("{} tasks total should take about {} minutes with 8 threads at 3.5GHz.".format(
           num_tasks, num_tasks / 35 * NPMI_PART_SIZE / 100))
    make_npmi_dir()
    voc_range = np.arange(voc_size)
    Parallel(n_jobs=-1, verbose=10)(delayed(build_and_save_npmi_submatrix)(batch)
                                    for batch in np.split(voc_range, voc_range[NPMI_PART_SIZE::NPMI_PART_SIZE]))



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--dir", nargs="?", default="./txt/", help="Path to the directory with documents")
    dir_path = arg_parser.parse_args().dir

    index, doc_id_to_path = build_index()
    vocabulary, term_to_id = get_vocabulary()
    voc_size = vocabulary.shape[0]
    inverted_index = build_inverted_index()
    num_docs = get_documents_number()
    idfs = get_inverse_document_frequencies()
    tfidf_matrix = get_tfidf_matrix()
    npmi_matrix = build_and_save_npmi_matrix()

    print("Saving the other results... ", end=""), sys.stdout.flush()
    start_time = clock()
    cPickle.dump((term_to_id, inverted_index, doc_id_to_path), open(PICKLE_PATH, "wb"))
    np.save(TFIDF_PATH, tfidf_matrix)
    np.save(VOCABULARY_PATH, vocabulary)
    print("{:.0f} s".format(clock() - start_time))
