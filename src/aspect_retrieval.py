# Takes the final aspect embedding matrix and then gets the aspects form itself.
from gensim.models import Word2Vec
import pickle
import numpy as np

def retrieve_aspects(aspects, num):
    aspects = aspects.detach().numpy()
    model = Word2Vec.load('model.bin')
    f = open("../results.txt","w")
    for i in aspects:
        sim = model.similar_by_vector(i, topn=num)
        f.write(str(sim))
        f.write("\n")
        print(sim)

if __name__ == '__main__':
    infile = open('../aspects_emeddings.pickle','rb')
    aspects = pickle.load(infile)
    infile.close()

    retrieve_aspects(aspects, 1)
