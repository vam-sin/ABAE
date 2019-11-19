# Reconstructs the sentence
from scipy.special import softmax
import numpy as np

def calculate_pt(W, zs, b):
    return softmax(np.matmul(W, zs) + b)

def calculate_rs(T, W, zs, b):
    T = T.T
    pt = calculate_pt(W, zs, b)
    rs = np.matmul(T, pt)

    return rs
