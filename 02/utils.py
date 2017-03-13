from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize

PICKLE_PATH = "stuff.pkl"

def get_terms(text):
    return tokenize(text.lower())
