#!/usr/bin/env python
import os
import cPickle

from generative_lm import GenerativeLanguageModel


def read_corpus(dir_path):
    corpus = []
    for document_name in os.listdir(dir_path):
        if document_name.endswith(".txt"):
            document_path = os.path.join(dir_path, document_name)
            with open(document_path) as f:
                corpus.append(f.read())
    return corpus

print "Building language model..."
lm = GenerativeLanguageModel()
lm.fit(read_corpus("txt"))

print "Dumping the model..."
cPickle.dump(lm, open("lm", "wb"))
