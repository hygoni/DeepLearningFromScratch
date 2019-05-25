import numpy as np

def step_function(x):
    y = x > 0
    return y.astrype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    if x < 0:
        return 0
    else:
        return x

def mse(y, t):
    return np.sum( (y-t) *i* 2 ) / len(t)

def cee():
