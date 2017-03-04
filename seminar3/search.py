#!/usr/bin/env python
from __future__ import print_function
import cPickle
import os
from time import clock

import numpy as np
import sys
from tabulate import tabulate

from utils import get_terms, get_npmi_part_path, PICKLE_PATH, TFIDF_PATH, NPMI_PART_SIZE, VOCABULARY_PATH

TOP_SIZE = 10


def get_documents_ids(terms, inverted_index):
    doc_ids = set()
    for term in terms:
        doc_ids = doc_ids.union(inverted_index[term])
    return np.array(list(doc_ids))


def get_top_pairs(tfidf_subvector, npmi_subsubmatrix, top = 10):
    tuples = []
    npmi_subsubmatrix_abs = np.abs(npmi_subsubmatrix)
    q_terms_sub, d_terms_sub = np.unravel_index(npmi_subsubmatrix_abs.ravel().argsort()[::-1], npmi_subsubmatrix_abs.shape)
    for i in range(min(top, len(q_terms_sub))):
        q_term_sub = q_terms_sub[i]
        d_term_sub = d_terms_sub[i]
        tuples.append((q_term_sub, d_term_sub, tfidf_subvector[d_term_sub], npmi_subsubmatrix[q_term_sub, d_term_sub]))
    return tuples


def load_npmi_submatrix(query_terms):
    npmi_submatrix = []
    for term in query_terms:
        npmi_vector = np.load(get_npmi_part_path(term))[term % NPMI_PART_SIZE]
        npmi_submatrix.append(npmi_vector)
    return np.array(npmi_submatrix)


def main():
    print("Loading resources... ", end=""), sys.stdout.flush()
    start_time = clock()
    term_to_id, inverted_index, doc_id_to_path = cPickle.load(open(PICKLE_PATH, "rb"))
    tfidf_matrix = np.load(TFIDF_PATH)
    vocabulary = np.load(VOCABULARY_PATH)
    print("{:.0f} s".format(clock() - start_time))

    while True:
        query = raw_input("\nType your query: ")
        print()
        query_terms = get_terms(query)
        # Get id-s for terms presented in vocabulary
        query_terms = [term_to_id[t] for t in query_terms if t in term_to_id]
        if not len(query_terms):
            continue
        documents_ids = get_documents_ids(query_terms, inverted_index)

        tfidf_submatrix = tfidf_matrix[documents_ids, :]
        npmi_submatrix = load_npmi_submatrix(query_terms)
        npmi_sum = npmi_submatrix.sum(axis=0)  # sum NPMI vectors for query terms (due to q=1)
        similarities = (tfidf_submatrix * npmi_sum.reshape((1, -1))).sum(axis=1)
        ranked_order = similarities.argsort()[::-1]

        top_table = []
        for i in ranked_order[:TOP_SIZE]:
            doc_id = documents_ids[i]
            file_name = os.path.basename(doc_id_to_path[documents_ids[i]])

            tfidf_vector = tfidf_matrix[doc_id]
            doc_terms = np.where(tfidf_vector)[0]
            tfidf_subvector = tfidf_vector[doc_terms]
            npmi_subsubmatrix = npmi_submatrix[:, doc_terms]

            tuples = get_top_pairs(tfidf_subvector, npmi_subsubmatrix)
            tuples = map(lambda tup: (vocabulary[query_terms[tup[0]]], vocabulary[doc_terms[tup[1]]]) + tup[2:], tuples)

            top_table.append([similarities[i], file_name] + tuples[:1])
            for tup in tuples[1:]:
                top_table.append(["", "", tup])
            top_table.append(["", "", ""])

        print(tabulate(top_table, showindex=False, headers=["sim", "doc", "(u, v, v_tfidf, uv_npmi)"] + query_terms, numalign="left"))


if __name__ == "__main__":
    main()
