#!/usr/bin/env python
import cPickle
import os

import numpy as np
from tabulate import tabulate

from utils import get_terms
from make_index import PICKLE_PATH

TOP_SIZE = 10


def get_documents_ids(terms, inverted_index):
    doc_ids = set()
    for term in terms:
        doc_ids = doc_ids.union(inverted_index.get(term, set()))
    return np.array(list(doc_ids))


def get_tfidf_matrix(documents_ids, terms, inverted_index, term_to_idf):
    tfidf_matrix = np.array([[0.] * len(terms) for _ in range(len(documents_ids))])
    doc_id_to_position = dict(zip(documents_ids, range(len(documents_ids))))
    for term_id, term in enumerate(terms):
        if term not in term_to_idf:
            continue
        idf = term_to_idf[term]
        for doc_id, tf in inverted_index[term].iteritems():
            tfidf_matrix[doc_id_to_position[doc_id]][term_id] = tf * idf
    return tfidf_matrix


def main():
    inverted_index, id_to_path, term_to_idf = cPickle.load(open(PICKLE_PATH, "rb"))

    while True:
        query = raw_input("\nType your query: ")
        print
        terms = get_terms(query)
        if not len(terms):
            continue
        documents_ids = get_documents_ids(terms, inverted_index)
        tfidf_matrix = get_tfidf_matrix(documents_ids, terms, inverted_index, term_to_idf)
        similarities = tfidf_matrix.sum(axis=1) / len(terms)

        # Sort arrays
        ranked_order = similarities.argsort()[::-1]
        similarities = similarities[ranked_order]
        documents_ids = documents_ids[ranked_order]
        tfidf_matrix = tfidf_matrix[ranked_order, :]

        top_table = []
        for i in range(min(TOP_SIZE, len(documents_ids))):
            file_name = os.path.basename(id_to_path[documents_ids[i]])
            top_table.append([similarities[i], file_name] + list(tfidf_matrix[i, :]))
        print tabulate(top_table, showindex=False, headers=["sim", "doc"] + terms, numalign="left")


if __name__ == "__main__":
    main()
