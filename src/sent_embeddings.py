from src.word2vec import get_word_embedding
import numpy as np
import math
import torch
import time
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def calculate_ys(sentence, E, unique_words):
    sentence = word_tokenize(sentence)
    process_sent = []
    for i in sentence:
        if i not in stop_words:
            process_sent.append(i)
    ys = 0
    for i in process_sent:
        ys += get_word_embedding(i, E, unique_words)

    return ys/len(process_sent)

def calculate_di(word, M, ys, E, unique_words):

    ew = get_word_embedding(word, E, unique_words)

    ew = np.asarray(ew)
    ew = torch.from_numpy(ew)
    ys = np.asarray(ys)
    ys = torch.from_numpy(ys)
    # print(ew)
    # ew = ew.t()
    # print(ew)
    interm = torch.matmul(ew, M)

    return torch.matmul(interm, ys)

def calculate_ai(word, sentence, M, ys, E, unique_words):
    num = math.exp(calculate_di(word, M, ys, E, unique_words))
    denom = 0
    for i in sentence:
        denom+= math.exp(calculate_di(i, M, ys, E, unique_words))

    return num/denom

def calculate_zs(sentence, M, ys, E, unique_words):
    sentence = word_tokenize(sentence)
    process_sent = []
    for i in sentence:
        if i not in stop_words:
            process_sent.append(i)
    zs = 0
    for word in process_sent:
        zs+= calculate_ai(word, process_sent, M, ys, E, unique_words) * get_word_embedding(word, E, unique_words)

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

    ys = calculate_ys(reviews[1], E, unique_words)

    print(calculate_zs(reviews[1], M, ys, E, unique_words))

    print(time.time() - start)
