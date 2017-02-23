from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize


def get_terms(text):
    return map(str.lower, tokenize(text))
