import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from src.sent_embeddings import calculate_ys, calculate_zs
from src.sent_reconstruction import calculate_rs
from src.hinge_loss import regularized_loss_value
from torch.autograd import Variable

# Tasks
# 1. Use the functions to get the aspects. Done
# 2. Reconstruct the sentence. Done
# 3. Convert to tensors.
# 4. Loss function optimization. Working

# Hyperparameters
epochs = 15
l = 1 # lambda hyperparameter
batch_size = 50
lr = 0.001 # learning rate
m = 20 # negative samples for each input

# Model
class ABAE_Model(nn.Module):

    def __init__(self, M, E, unique_words, T, W, b):
        super(ABAE_Model, self).__init__()
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
        rs = calculate_rs(self.T, self.W, zs, self.b)

        return rs, ys, zs


# Parameters
M = Variable(torch.randn(200, 200).type(torch.float32), requires_grad=True)
W = Variable(torch.randn(200, 200).type(torch.float32), requires_grad=True)
b = Variable(torch.randn(200, 1).type(torch.float32), requires_grad=True)

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
T = np.asarray(T)
T = torch.tensor(T, requires_grad=True)

# Model Instantiation
abae = ABAE_Model(M, E, unique_words, T, W, b)

# Training of the model
for i in range(epochs):
    epoch_loss = 0.0
    print("Epoch Number: " + str(i))
    for j in range(1, len(reviews)):
        rs, ys, zs = abae(reviews[j])
        loss = regularized_loss_value(rs, zs, ys, T, l)
        epoch_loss += loss

        print(loss)
        loss.backward()

        M.data -= lr * M.grad.data
        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        T.data -= lr * T.grad.data

        M.grad.data.zero_()
        W.grad.data.zero_()
        b.grad.data.zero_()
        T.grad.data.zero_()

    print("Epoch Loss: " + str(epoch_loss) +"\n")
