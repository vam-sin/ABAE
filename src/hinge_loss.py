# calculates hinge loss
# l is the lamda hyperparameter

import numpy as np
import torch

def calculate_jtheta(rs, zs, ys):
    # ni is same as ys
    # This is for one sentence
    val = 1 - (rs*zs) + (rs*ys)
    mean = torch.mean(val)

    return max(0, mean)

def calculate_utheta(T):
    return (torch.matmul(T, T.t()) - torch.eye(14).type(torch.DoubleTensor))

def regularized_loss_value(rs, zs, ys, T, l):
    ys = torch.from_numpy(ys)
    zs = torch.from_numpy(zs)
    ys = ys.type(torch.DoubleTensor)
    zs = zs.type(torch.DoubleTensor)

    return calculate_jtheta(rs, zs, ys) + (l*calculate_utheta(T))
