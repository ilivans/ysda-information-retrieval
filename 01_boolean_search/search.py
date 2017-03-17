#!/usr/bin/env python
import cPickle
from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize


def main():
    inverted_index, id_to_path = cPickle.load(open("iindex.pkl", "rb"))

    while True:
        query = raw_input("\nType your query: ")
        words = map(str.lower, tokenize(query))
        documents_ids = set()
        for i, word in enumerate(words):
            if word not in inverted_index:
                documents_ids = set()
                break
            documents_ids = documents_ids & inverted_index[word] if i else inverted_index[word]
        document_paths = map(lambda doc_id: id_to_path[doc_id], list(documents_ids)[:10])
        print "\n".join(document_paths)


if __name__ == "__main__":
    main()
