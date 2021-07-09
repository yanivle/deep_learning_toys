import numpy as np
from numpy.testing import assert_almost_equal
import diff_util

def softmax_naive(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmax_stable(x):
    p = np.exp(x - np.max(x))
    return p / np.sum(p)


softmax = softmax_stable

assert_almost_equal(softmax(np.array([range(3)])), [[0.09003057, 0.24472847, 0.66524096]])


def softmax_derivative(ys, target: int):
    res = np.copy(ys)
    res[0, target] -= 1
    return res


def softmax_loss(os, target: int):
    return -np.log(softmax(os)[0, target])


os = np.array([[1., 2., 3.]])
diff_util.gradcheck(softmax_loss, (os, 1), [os], [softmax_derivative(softmax(os), 1)], ['os'])
