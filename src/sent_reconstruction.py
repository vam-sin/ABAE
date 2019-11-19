# Reconstructs the sentence
from torch.nn import Softmax
import numpy as np
import torch

def calculate_pt(W, zs, b):
    interm = torch.matmul(W, zs) + b
    m = Softmax(dim = 0)

    return m(interm)

def calculate_rs(T, W, zs, b):
    zs = torch.from_numpy(zs)
    # T = T.t()
    pt = calculate_pt(W, zs, b)
    pt = pt.type(torch.DoubleTensor)
    rs = torch.matmul(T, pt)

    return rs
