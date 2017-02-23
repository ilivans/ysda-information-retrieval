from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize


def get_terms(text):
    return tokenize(text.lower())
