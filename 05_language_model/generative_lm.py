from collections import defaultdict
from heapq import heappushpop, heappush
from random import choice

from lm import LanguageModel


class GenerativeLanguageModel(LanguageModel):

    def __init__(self, n=3, lambd=0.95, diversity=20, seq_len=15):
        super(GenerativeLanguageModel, self).__init__(n, lambd)
        self._diversity = diversity
        self._seq_len = seq_len
        self._predictor = defaultdict(lambda: [])  # maps n-gram to the most suitable variants of the next term
        self._generator = None

    def fit(self, corpus):
        super(GenerativeLanguageModel, self).fit(corpus)
        self._build_predictor()
        self._lms = None

    def _build_predictor(self):
        for n in range(1, self._n + 1):
            for n_gram, prob in self._lms[n].iteritems():
                prefix, term = n_gram[:-1], n_gram[-1]
                if len(self._predictor[prefix]) < self._diversity:
                    heappush(self._predictor[prefix], (prob, term))
                else:
                    heappushpop(self._predictor[prefix], (prob, term))
        self._predictor = dict(self._predictor)

    def generate(self, prefix):
        prefix = self._words_encoder.transform(prefix)
        prefix_len = self._n - 1
        if len(prefix) > prefix_len:
            raise ValueError("Incorrect input sequence's length: got {}, required no more than {}.".format(len(prefix),
                                                                                                          prefix_len))
        # Get rid of unknown words
        for i, t in enumerate(prefix):
            if t is None:
                prefix = prefix[i + 1:]
        prefix = tuple(prefix)
        for _ in xrange(self._seq_len - self._n + 1):
            term = self._generate(prefix)
            yield self._words_encoder.inverse_transform([term])
            prefix = prefix + (term,) if len(prefix) < prefix_len else prefix[1:] + (term,)

    def _generate(self, prefix):
        while True:
            terms = self._predictor.get(prefix, None)
            if terms is None:
                prefix = prefix[1:]
                continue
            return choice(terms)[1]
