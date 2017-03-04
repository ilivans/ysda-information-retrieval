from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize

PICKLE_PATH = "stuff.pkl"
TFIDF_PATH = "tfidf.npy"
NPMI_PATH = "npmi.npy"

def get_terms(text):
    return tokenize(text.lower())
