import os

from nltk.tokenize import RegexpTokenizer

tokenize = RegexpTokenizer("\w+").tokenize

PICKLE_PATH = "stuff.pkl"
TFIDF_PATH = "tfidf.npy"
VOCABULARY_PATH = "vocabulary.npy"

NPMI_FOLDER = "npmi_parts"
NPMI_PATH_TEMPLATE = os.path.join(NPMI_FOLDER, "npmi{}-{}.npy")
NPMI_PART_SIZE = 100

def get_terms(text):
    return tokenize(text.lower())


def make_npmi_dir():
    if not os.path.exists(NPMI_FOLDER):
        os.mkdir(NPMI_FOLDER)


def get_npmi_part_path(term):
    left_border = term / NPMI_PART_SIZE * NPMI_PART_SIZE
    return NPMI_PATH_TEMPLATE.format(left_border, left_border + NPMI_PART_SIZE - 1)
