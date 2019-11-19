# calculates hinge loss
import numpy as np

def calculate_jtheta(rs, zs, ys):
    # ni is same as ys
    # This is for one sentence
    val = 1 - (rs*zs) + (rs*ys)
    return max(0, val)

def calculate_utheta(T):
    return (np.matmul(T, T.T) - np.identity(14, dtype = float))

def regularized_loss_value(rs, zs, ys, T, lambda):
    return calculate_jtheta(rs, zs, ys) + (lambda*calculate_utheta(T))
