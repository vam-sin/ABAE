import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

# Tasks
# 1. Use the functions to get the aspects.
# 2. Reconstruct the sentence.
# 3. Loss function optimization.

# Hyperparameters

# Model
class Model(nn.Module):

    def __init__(self, M, E, unique_words, T, W, b):
        self.M = M
        self.E = E
        self.unique_words = unique_words
        self.T = T
        self.W = W
        self.b = b

    def forward(self, input):
        # Forward pass of the review given
        ys = calculate_ys(input, self.E, self.unique_words)
        zs = calculate_zs(input, self.M, ys, self.E, self.unique_words)
        rs = calculate_rs(self.T, self.W, zs, self.B)


# Parameters
M = np.random.randn(200, 200)
W = np.random.randn(200, 200)
W = np.random.randn(200, 1)

infile = open('src/E.pickle','rb')
E = pickle.load(infile)
infile.close()

infile = open('src/reviews.pickle','rb')
reviews = pickle.load(infile)
infile.close()

infile = open('src/unique_words.pickle','rb')
unique_words = pickle.load(infile)
infile.close()

infile = open('src/T.pickle','rb')
T = pickle.load(infile)
infile.close()
