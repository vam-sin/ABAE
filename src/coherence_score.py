# Calculates coherence score given the aspect_embeddings pickle file.

import numpy as np
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
from gensim.models import Word2Vec

def D1(word, reviews):
    # Document frequency of word
    doc_freq = 0
    for rev in reviews:
        rev = word_tokenize(rev)
        if word in rev:
            doc_freq += 1

    return doc_freq

def D2(word1, word2, reviews):
    # codocument frequency
    doc_freq = 0
    for rev in reviews:
        rev = word_tokenize(rev)
        if word1 in rev and word2 in rev:
            doc_freq += 1

    return doc_freq

def create_sz(word_vec, n):
    # Top n words
    model = Word2Vec.load('model.bin')
    sim = model.similar_by_vector(word_vec, topn = n)

    sz = []
    for i in sim:
        sz.append(i[0])

    return sz

def coherence_score(aspects_vec, n, reviews):
    print("Calculating Coherence Score.")
    cs = 0.0
    for i in aspects_vec:
        word_score = 0.0
        sz = create_sz(i, n)
        for j in range(1, n):
            for k in range(j):
                word_score += np.log10((D2(sz[j], sz[k], reviews)+1) / D1(sz[k], reviews))
        # print(word_score)
        cs += word_score
    cs = cs/len(aspects_vec)

    print("Coherence Score is: ", cs)

    return cs

if __name__ == '__main__':

    infile = open('../aspects_emeddings.pickle','rb')
    aspects = pickle.load(infile)
    infile.close()

    infile = open('reviews.pickle','rb')
    reviews = pickle.load(infile)
    infile.close()

    aspects = aspects.detach().numpy()
    coherence_score(aspects, 5, reviews)
