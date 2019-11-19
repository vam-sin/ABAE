# Takes the final aspect embedding matrix and then gets the aspects form itself.
from gensim.models import Word2Vec
import pickle
import numpy as np

def retrieve_aspects(aspects):
    aspects = aspects.detach().numpy()
    model = Word2Vec.load('model.bin')
    for i in aspects:
        print(model.similar_by_vector(i, topn=1))

if __name__ == '__main__':
    infile = open('../aspects_emeddings.pickle','rb')
    aspects = pickle.load(infile)
    infile.close()

    retrieve_aspects(aspects)
