import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Tasks
# 1. Use the functions to get the aspects.
# 2. Reconstruct the sentence.
# 3. Loss function optimization.

# Hyperparameters

# Train
infile = open('src/reviews.pickle','rb')
reviews = pickle.load(infile)
infile.close()
