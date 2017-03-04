#!/usr/bin/env python
import cPickle
import os

import numpy as np
from tabulate import tabulate

from utils import get_terms, PICKLE_PATH, TFIDF_PATH, NPMI_PATH

TOP_SIZE = 10


def get_documents_ids(terms, inverted_index):
    doc_ids = set()
    for term in terms:
        doc_ids = doc_ids.union(inverted_index.get(term, set()))
    return np.array(list(doc_ids))


def get_top_pairs(tfidf_subvector, npmi_subsubmatrix, k = 10):
    tuples = []
    npmi_subsubmatrix_abs = np.abs(npmi_subsubmatrix)
    q_terms_sub, d_terms_sub = np.unravel_index(npmi_subsubmatrix_abs.ravel().argsort(), npmi_subsubmatrix_abs.shape)
    for i in range(min(k, len(q_terms_sub))):
        q_term_sub = q_terms_sub[i]
        d_term_sub = d_terms_sub[i]
        tuples.append((q_term_sub, d_term_sub, tfidf_subvector[d_term_sub], npmi_subsubmatrix[q_term_sub, d_term_sub]))
    return tuples


def main():
    vocabulary, term_to_id, inverted_index, doc_id_to_path = cPickle.load(open(PICKLE_PATH, "rb"))
    tfidf_matrix = np.load(TFIDF_PATH)
    npmi_matrix = np.load(NPMI_PATH)

    while True:
        query = raw_input("\nType your query: ")
        print
        query_terms = get_terms(query)
        # Get id-s for terms presented in vocabulary
        query_terms = [term_to_id[t] for t in query_terms if t in term_to_id]
        if not len(query_terms):
            continue
        documents_ids = get_documents_ids(query_terms, tfidf_matrix)

        tfidf_submatrix = tfidf_matrix[documents_ids, :]
        npmi_submatrix = npmi_matrix[query_terms, :]
        npmi_sum = npmi_submatrix.sum(axis=0)  # sum NPMI vectors for query terms
        similarities = (tfidf_submatrix * npmi_sum.reshape((1, -1))).sum(axis=1)

        # Sort arrays
        ranked_order = similarities.argsort()[::-1]
        # similarities = similarities[ranked_order]
        # documents_ids = documents_ids[ranked_order]
        # tfidf_matrix = tfidf_matrix[ranked_order, :]

        top_table = []
        for i in ranked_order[:TOP_SIZE]:
            doc_id = documents_ids[i]
            file_name = os.path.basename(doc_id_to_path[documents_ids[i]])

            tfidf_vector = tfidf_matrix[doc_id]
            doc_terms = tfidf_vector.argwhere()
            tfidf_subvector = tfidf_vector[doc_terms]
            npmi_subsubmatrix = npmi_submatrix[:, doc_terms]

            tuples = get_top_pairs(tfidf_subvector, npmi_subsubmatrix)
            tuples = map(lambda q_term_sub, d_term_sub, *other:
                        (vocabulary[query_terms[q_term_sub]], vocabulary[doc_terms[d_term_sub]]) + other,
                        tuples)

            top_table.append([similarities[i], file_name] + tuples[:1])
            for tup in tuples[1:]
                top_table.append(["", "", tup])

        print tabulate(top_table, showindex=False, headers=["sim", "doc"] + query_terms, numalign="left")


if __name__ == "__main__":
    main()
