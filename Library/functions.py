import numpy as np
import sys, copy
sys.path.append('.')

'''
Cost functions
'''

def mean_squared_error(y_predict, y_test):
    return 0.5 * np.sum((y_predict - y_test) ** 2)


def mse(y_predict, y_test):
    return mean_squared_error(y_predict, y_test)


def cross_entropy_error(y_predict, y_test):
    if y_predict.ndim == 1:
        y_test = y_test.reshape(1, y_test.size)
        y_predict = y_predict.reshape(1, y_predict.size)

    if y_test.size == y_predict.size:
        y_test = y_test.argmax(axis=1)

    batch_size = y_predict.shape[0]
    return -np.sum(np.log(y_predict[np.arange(batch_size), y_test] + 1e-7)) / batch_size


def cee(y_predict, y_test):
    return cross_entropy_error(y_predict, y_test)

'''
Activation functions
'''
def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    C = np.max(x) #Preventing overflow
    exp_x = np.exp(x - C)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

'''
Calculus & Numerical Gradients
'''


def numerical_diff(f_, x):  # f: function, x : x where we want to get differential coefficient
    h = 10e-4
    return (f_(x + h) - f_(x - h)) / (2 * h)


def diff(f_, x):
    return numerical_diff(f_, x)


def numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1d(f, x)

        return grad