import numpy as np
from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize


def get_terms(text):
    return tokenize(text.lower())


class WordsEncoder:

    def __init__(self, max_len=None):
        self.term_to_id = dict()
        self.vocabulary = []
        self.voc_size = 0

    def _terms_to_ids(self, seq):
        ids = [self.term_to_id.get(t, None) for t in seq]
        return np.array(ids)

    def fit(self, text):
        terms = get_terms(text)
        self.vocabulary = list(set([term for term in terms]))
        self.voc_size = len(self.vocabulary)
        self.term_to_id = dict(zip(self.vocabulary, range(len(self.vocabulary))))
        return self

    def transform(self, text):
        return self._terms_to_ids(get_terms(text))

    def fit_transform(self, text, y=None):
        self.fit(text)
        return self.transform(text)

    def inverse_transform(self, terms):
        return " ".join([self.vocabulary[t] for t in terms])
