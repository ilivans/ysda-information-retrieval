from collections import Counter

import numpy as np

from words_encoder import WordsEncoder


class LanguageModel(object):
    def __init__(self, n=3, lambd=0.95):
        self._n = n  # size of n-gram
        self._lambd = lambd
        self._corpus_len = 0
        self._words_encoder = None
        self._corpus = None
        self._lms = [{(): 1.}]

    def get_n(self):
        return self._n

    def fit(self, corpus):
        if isinstance(corpus, list):
            corpus = " ".join(corpus)
        self._words_encoder = WordsEncoder()
        self._corpus = self._words_encoder.fit_transform(corpus)
        self._corpus_len = self._corpus.shape[0]

        for n in range(1, self._n + 1):
            self._build_lm(n)
        self._corpus = None

    def _build_lm(self, n):
        denominator = float(self._corpus_len - n + 1)
        n_grams = np.array([self._corpus[i:self._corpus_len - n + i + 1] for i in range(n)]).T
        n_grams = map(tuple, n_grams)
        lm_n = dict(Counter(n_grams))
        for t, c in lm_n.iteritems():
            lm_n[t] /= denominator
            # Smoothing
            if n > 1:
                lm_n[t] = self._lambd * lm_n[t] + (1 - self._lambd) * self._lms[-1][t[1:]]
        self._lms.append(lm_n)
