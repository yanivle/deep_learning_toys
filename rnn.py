import numpy as np
import diff_util
from numpy.testing import assert_almost_equal
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)


assert_almost_equal(sigmoid_derivative(sigmoid(4)), diff_util.differentiate_at(sigmoid, 4))


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


class RNN:
    def __init__(self, vocab_size: int, hidden_dim: int, epsilon_init: float = 1e-5):
        # x, h, and y are all column vectors
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.Wxh = np.random.randn(hidden_dim, vocab_size) * epsilon_init
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * epsilon_init
        self.Why = np.random.randn(vocab_size, hidden_dim) * epsilon_init
        self.bh = np.zeros(self.h_shape)
        self.by = np.zeros(self.y_shape)
        self.params = (self.Wxh, self.Whh, self.Why, self.bh, self.by)
        self.params_names = ('Wxh', 'Whh', 'Why', 'bh', 'by')

        # For rprop and adagrad:
        self.dparams_prev = tuple(np.zeros_like(param)
                                  for param in self.params)
        self.step_multiplers = tuple(np.ones_like(param)
                                     for param in self.params)

    def __repr__(self) -> str:
        return 'RNN(' + ', '.join([f'{name}:{param}' for param, name in zip(self.params, self.params_names)]) + ')'

    def onehot(self, x: int) -> np.array:
        res = np.zeros((self.vocab_size, 1))
        res[x] = 1
        return res

    @property
    def x_shape(self): return (self.vocab_size, 1)

    @property
    def h_shape(self): return (self.hidden_dim, 1)

    @property
    def y_shape(self): return (self.vocab_size, 1)

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
            h = np.zeros(self.h_shape)
        xs, hs, ys = {}, {-1: h}, {}
        for t, input in enumerate(inputs):
            x, h, y = self.forward_one_step(input, h)
            xs[t], hs[t], ys[t] = x, h, y
        return xs, hs, ys

    def sample(self, prefix, length, randomize=False):
        res = [prefix[0]]
        h = np.zeros(self.h_shape)
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
        dhnext = np.zeros(self.h_shape)
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

    def update_rprop(self, dparams, learning_rate):
        for param, dparam, dparam_prev, step_multiplier in zip(self.params, dparams, self.dparams_prev, self.step_multiplers):
            step_multiplier[np.sign(dparam) == np.sign(dparam_prev)] *= 1.2
            step_multiplier[np.sign(dparam) != np.sign(dparam_prev)] /= 2
            np.clip(step_multiplier, 0.01, 100, out=step_multiplier)
            np.copyto(dparam_prev, dparam)
            param -= dparam * step_multiplier * learning_rate

    def update_adagrad(self, dparams, learning_rate):
        for param, dparam, dparam_prev in zip(self.params, dparams, self.dparams_prev):
            dparam_prev += dparam * dparam
            param -= dparam * learning_rate / np.sqrt(dparam_prev + 1e-8)

    def update(self, dparams, learning_rate):
        for param, dparam in zip(self.params, dparams):
            param -= dparam * learning_rate

    def train(self,
              training_set,
              iters: int = 1_000,
              learning_rate: float = 1e-1,
              max_pass_length: int = 32,
              optimizer_name: str = 'adagrad',
              print_every: int = 100,
              sample_prefix_length: int = 1,
              sample_length: int = 100,
              sample_tokenizer=str,
              randomize_sample=False,
              plot_every=None,
              plot_color='b'):
        optimizer = {
            'gd': self.update,
            'rprop': self.update_rprop,
            'adagrad': self.update_adagrad,
        }[optimizer_name]
        loss_history = []
        smooth_loss = -np.log(1.0 / self.vocab_size) * sum(len(targets) for _, targets in training_set)
        if plot_every:
            plt.ylabel('Loss')
            plt.show(block=False)
        i = 0
        while True:
            for inputs, targets in training_set:
                h = None
                for start in range(0, len(inputs), max_pass_length):
                    if i % print_every == 0:
                        inputs, targets = random.choice(training_set)
                        sampled = self.sample(
                            inputs[:sample_prefix_length], sample_length, randomize_sample)
                        print('\n' + ''.join([sample_tokenizer(x) for x in sampled]))
                    xs, hs, ys = self.forward(
                        inputs[start:start + max_pass_length], h)
                    h = hs[len(hs) - 2]
                    loss = self.loss(ys, targets[start:start + max_pass_length])
                    dparams = self.backward(xs, hs, ys, targets[start:start + max_pass_length])
                    smooth_loss = smooth_loss * 0.99 + loss * 0.01
                    loss_history.append(smooth_loss)
                    optimizer(dparams, learning_rate)
                    if i % print_every == 0:
                        print(i, 'Loss:', smooth_loss)
                    if plot_every and i % plot_every == 0:
                        plt.plot(loss_history, plot_color)
                        plt.pause(0.05)
                    i += 1
