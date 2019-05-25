import numpy as np

def perceptron(inputs, weights, bias):
    tmp = np.sum(inputs * weights) + bias
    if tmp <= 0:
        return 0
    else:
        return 1


