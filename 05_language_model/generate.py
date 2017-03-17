from __future__ import print_function
import cPickle

print("Loading the model...")
lm = cPickle.load(open("lm", "rb"))

while True:
    query = raw_input("\nType {} words: ".format(lm.get_n() - 1))
    print(query, end=" ")
    try:
        for term in lm.generate(query):
            print(term, end=" ")
    except ValueError as e:
        print()
        print(e)
    print()
