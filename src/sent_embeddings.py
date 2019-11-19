from word2vec import get_word_embedding
import numpy as np
import math
import time
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def calculate_ys(sentence, E, unique_words):
    ys = 0
    for i in sentence:
        ys += get_word_embedding(i, E, unique_words)

    return ys/len(sentence)

def calculate_di(word, M, ys, E, unique_words):

    ew = get_word_embedding(word, E, unique_words)

    ew = np.asarray(ew)
    M = np.asarray(M)
    ys = np.asarray(ys)

    ew = ew.T

    interm = np.matmul(ew, M)

    return np.matmul(interm, ys)

def calculate_ai(word, sentence, M, ys, E, unique_words):
    num = math.exp(calculate_di(word, M, ys, E, unique_words))
    denom = 0
    for i in sentence:
        denom+= math.exp(calculate_di(i, M, ys, E, unique_words))

    return num/denom

def calculate_zs(sentence, M, ys, E, unique_words):
    zs = 0
    for word in sentence:
        zs+= calculate_ai(word, sentence, M, ys, E, unique_words) * get_word_embedding(word, E, unique_words)

    return zs

if __name__ == '__main__':
    M = np.random.randn(200, 200)

    infile = open('unique_words.pickle','rb')
    unique_words = pickle.load(infile)
    infile.close()
    # print(unique_words)
    # print(unique_words.index('charge'))
    infile = open('reviews.pickle','rb')
    reviews = pickle.load(infile)
    infile.close()

    infile = open('E.pickle','rb')
    E = pickle.load(infile)
    infile.close()

    start = time.time()
    sent = word_tokenize(reviews[1])
    process_sent = []
    for i in sent:
        if i not in stop_words:
            process_sent.append(i)

    ys = calculate_ys(process_sent, E, unique_words)

    print(calculate_zs(process_sent, M, ys, E, unique_words))

    print(time.time() - start)
