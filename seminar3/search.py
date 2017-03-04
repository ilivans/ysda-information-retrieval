#!/usr/bin/env python
import cPickle
import os

import numpy as np
from tabulate import tabulate

from utils import get_terms, PICKLE_PATH, TFIDF_PATH, NPMI_PATH

TOP_SIZE = 10


def get_documents_ids(terms, tfidf_matrix):
    tfidf_sums = np.zeros(tfidf_matrix.shape[0])
    for term in terms:
        tfidf_sums += tfidf_matrix[:, term]
    doc_ids = np.argwhere(tfidf_sums)  # get documents that have non-zero tfidf sum for terms
    return doc_ids


def main():
    vocabulary, term_to_id, inverted_index, doc_id_to_path = cPickle.load(open(PICKLE_PATH, "rb"))
    tfidf_matrix = np.load(TFIDF_PATH)
    npmi_matrix = np.load(NPMI_PATH)

    while True:
        query = raw_input("\nType your query: ")
        print
        terms = get_terms(query)
        # Get id-s for terms presented in vocabulary
        terms = [term_to_id[t] for t in terms if t in term_to_id]
        if not len(terms):
            continue
        documents_ids = get_documents_ids(terms, tfidf_matrix)
        print documents_ids
        break
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
