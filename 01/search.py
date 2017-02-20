#!/usr/bin/env python
import cPickle
from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize


def main():
    inverted_index = cPickle.load(open("iindex.pkl", "rb"))

    while True:
        query = raw_input("\nType your query:\n")
        docs = set()
        for i, word in enumerate(tokenize(query)):
            word = word.lower()
            if word not in inverted_index:
                docs = set()
                break
            if not i:
                docs = inverted_index[word]
            else:
                docs = docs & inverted_index[word]
        print "\n".join(list(docs)[:10])


if __name__ == "__main__":
    main()
