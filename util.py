import numpy as np


def softmax_naive(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmax_stable(x):
    p = np.exp(x - np.max(x))
    return p / np.sum(p)


softmax = softmax_stable


def softmax_loss_derivative(ys, target: int):
    res = np.copy(ys)
    res[0, target] -= 1
    return res


def softmax_loss(os, target: int):
    return -np.log(softmax(os)[0, target])


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)
