from optimizers import AdaGrad, Optimizer
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from util import softmax


class RNN(Model):
    def __init__(self, vocab_size: int, hidden_dim: int, epsilon_init: float = 1e-5):
        super().__init__()
        # x, h, and y are all column vectors
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.addParam('Wxh', np.random.randn(hidden_dim, vocab_size) * epsilon_init)
        self.addParam('Whh', np.random.randn(hidden_dim, hidden_dim) * epsilon_init)
        self.addParam('Why', np.random.randn(vocab_size, hidden_dim) * epsilon_init)
        self.addParam('bh', np.zeros((self.hidden_dim, 1)))
        self.addParam('by', np.zeros((self.vocab_size, 1)))

    def onehot(self, x: int) -> np.array:
        res = np.zeros((self.vocab_size, 1))
        res[x] = 1
        return res

    def forward_one_step(self, input, hprev):
        assert np.isscalar(input), (input, type(input))
        x = self.onehot(input)
        z = np.dot(self.Wxh, x) + np.dot(self.Whh, hprev) + self.bh
        h = np.tanh(z)
        o = np.dot(self.Why, h) + self.by
        y = softmax(o)
        return x, h, y

    def forward(self, inputs, h=None):
        if h is None:
            h = np.zeros((self.hidden_dim, 1))
        xs, hs, ys = {}, {-1: h}, {}
        for t, input in enumerate(inputs):
            x, h, y = self.forward_one_step(input, h)
            xs[t], hs[t], ys[t] = x, h, y
        return xs, hs, ys

    def sample(self, prefix, length, randomize=False):
        res = [prefix[0]]
        h = np.zeros((self.hidden_dim, 1))
        prev_output = prefix[0]
        for i in range(length):
            _, h, y = self.forward_one_step(prev_output, h)
            if i + 1 < len(prefix):
                prev_output = prefix[i + 1]
            elif randomize:
                prev_output = np.random.choice(
                    range(self.vocab_size), p=y.ravel())
            else:
                prev_output = np.argmax(y)
            res.append(prev_output)
        return res

    def loss(self, ys, targets):
        assert len(ys) == len(targets)
        return np.sum(-np.log(ys[t][targets[t], 0] + np.spacing(0)) for t in range(len(ys)))

    def forward_loss(self, inputs, targets):
        _, _, ys = self.forward(inputs)
        return self.loss(ys, targets)

    def backward(self, xs, hs, ys, targets):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dparams = (dWxh, dWhh, dWhy, dbh, dby)
        dhnext = np.zeros((self.hidden_dim, 1))
        for t in reversed(range(len(targets))):
            do = np.copy(ys[t])
            do[targets[t]] -= 1
            dWhy += np.dot(do, hs[t].T)
            dby += do
            dh = np.dot(self.Why.T, do) + dhnext
            dz = (1 - hs[t] * hs[t]) * dh
            dbh += dz
            dWxh += np.dot(dz, xs[t].T)
            dWhh += np.dot(dz, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dz)
        for dparam in dparams:
            np.clip(dparam, -5, 5, out=dparam)
        return dparams

