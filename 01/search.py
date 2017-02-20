import cPickle
from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer('\w+').tokenize


if __name__=='__main__':
    iindex = cPickle.load(open('indexer.pkl', 'rb'))
    while True:
        query = raw_input('Type in a query')

    docs = set()
    for i, w in enumerate(tokenize(query)):
        w = w.lower()
        if not i:
            docs = iindex[w]
        else:
            docs = docs & iindex[w]
    for j, d in enumerate(docs):
        if j < 10:
            print d
        else:
            break
