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

def regularized_loss_value(list_rs, list_zs, list_ys, T, l):
    # ni is same as yi
    jtheta = 0.0
    for i in range(len(list_rs)):
        ys = torch.from_numpy(list_ys[i])
        zs = torch.from_numpy(list_zs[i])
        ys = ys.type(torch.DoubleTensor)
        zs = zs.type(torch.DoubleTensor)

        jtheta += calculate_jtheta(list_rs[i], zs, ys)

    return jtheta + (l*calculate_utheta(T))
