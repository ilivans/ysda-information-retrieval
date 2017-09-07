import os

import numpy as np
import scipy.stats as sps

from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize


_DIR_PATH = "~/data/simple.wiki"

assert os.path.exists(_DIR_PATH)

Xs = []
simhashes = []
simhash_to_name = {}
for filename in os.listdir(_DIR_PATH):
    filepath = os.path.join(_DIR_PATH, filename)
    with open(filepath) as f:
        doc = f.read()

    X = np.zeros(64, dtype=np.int32)
    for hash_ in map(hash, tokenize(doc)):
        for idx, bit in enumerate(reversed(format(hash_, "b"))):
            X[idx] += 1 if bit == "1" else -1
    Xs.append(X)

    simhash = np.uint64(int("".join(reversed(map(str, np.clip(np.sign(X), 0, 1)))), 2))
    simhashes.append(simhash)

    assert simhash not in simhash_to_name
    simhash_to_name[simhash] = filename

Xs = np.array(Xs)
simhashes = np.array(simhashes)
np.save("simhashes", simhashes)

stds = Xs.std(axis=0)
alpha = 3. / 64  # can be changed
for X in Xs:
    for x, std in zip(X, stds):
        sps.norm(x, alpha * std)
