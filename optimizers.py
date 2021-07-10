import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from models import Model
import itertools


def default_logging_function(epoch: int, i: int, loss_history: list[float]) -> None:
    print(f'Epoch: {epoch}, iteration: {i}, loss: {loss_history[-1]}')


class PlotLogger:
    def __init__(self, color='b') -> None:
        self.color = color
        plt.ylabel('Loss')
        plt.show(block=False)

    def __call__(self, epoch: int, i: int, loss_history: list[float]) -> None:
        plt.plot(loss_history, self.color)
        plt.pause(0.05)
        default_logging_function(epoch, i, loss_history)


class Optimizer:
    def __init__(self, model: Model, learning_rate: float = 1e-1, l2_decay: float = 1e-3) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.l2_decay = l2_decay

    def update(self, dparams) -> None:
        raise NotImplemented()

    def train(self,
              training_set,
              epochs: int = 100,
              mini_batch_size: int = 100,
              logging_function=default_logging_function,
              log_every=None):
        random.shuffle(training_set)
        loss_history = []
        smooth_loss = None
        i = 0
        for epoch in tqdm(range(epochs)):
            for batch_start in range(0, len(training_set), mini_batch_size):
                mini_batch = training_set[batch_start: batch_start + mini_batch_size]
                for inputs, targets in mini_batch:
                    activations, caches = self.model.forward(inputs)
                    loss = self.model.loss(activations, targets)
                    smooth_loss = smooth_loss * 0.99 + loss * 0.01 if smooth_loss is not None else loss
                    loss_history.append(smooth_loss)
                    dparams = self.model.backward(activations, caches, targets)
                    dparams_flat = itertools.chain.from_iterable(dparams)
                    self.update(dparams_flat)
                    if log_every and i % log_every == 0:
                        logging_function(epoch, i, loss_history)
                    i += 1
        return loss_history

    def train_autoregressive(self,
                             training_set,
                             iters: int = 1_000_000,
                             max_pass_length: int = 32,
                             print_every: int = None,
                             sample_prefix_length: int = 1,
                             sample_length: int = 100,
                             sample_tokenizer=str,
                             randomize_sample=False,
                             plot_every=None,
                             plot_color='b'):
        loss_history = []
        smooth_loss = -np.log(1.0 / self.model.vocab_size) * sum(len(targets) for _, targets in training_set)
        if plot_every:
            plt.ylabel('Loss')
            plt.show(block=False)
        i = 0
        with tqdm(total=iters) as pbar:
            while True:
                for inputs, targets in training_set:
                    h = None
                    for start in range(0, len(inputs), max_pass_length):
                        if print_every and i % print_every == 0:
                            inputs, targets = random.choice(training_set)
                            sampled = self.model.sample(
                                inputs[:sample_prefix_length], sample_length, randomize_sample)
                            print('\n' + ''.join([sample_tokenizer(x) for x in sampled]))
                        xs, hs, ys = self.model.forward(
                            inputs[start:start + max_pass_length], h)
                        h = hs[len(hs) - 2]

                        loss = self.model.loss(ys, targets[start:start + max_pass_length])
                        smooth_loss = smooth_loss * 0.99 + loss * 0.01
                        loss_history.append(smooth_loss)

                        dparams = self.model.backward(xs, hs, ys, targets[start:start + max_pass_length])
                        self.update(dparams)

                        if print_every and i % print_every == 0:
                            print(i, 'Loss:', smooth_loss)
                        if plot_every and i % plot_every == 0:
                            plt.plot(loss_history, plot_color)
                            plt.pause(0.05)
                        i += 1
                        if i >= iters:
                            return loss_history
                        pbar.update(1)


class GradientDescent(Optimizer):
    def __init__(self, model, learning_rate: float = 1e-1, l2_decay: float = 1e-3) -> None:
        super().__init__(model, learning_rate, l2_decay)

    def update(self, dparams) -> None:
        for param, dparam in zip(self.model.params, dparams):
            param *= (1 - self.l2_decay)
            param -= dparam * self.learning_rate


class Momentum(Optimizer):
    def __init__(self, model, learning_rate: float = 1e-3, momentum: float = 0.9, l2_decay: float = 1e-3) -> None:
        super().__init__(model, learning_rate, l2_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(param) for param in model.params]

    def update(self, dparams) -> None:
        for param, dparam, velocity in zip(self.model.params, dparams, self.velocities):
            velocity *= self.momentum
            velocity -= self.learning_rate * dparam
            param *= (1 - self.l2_decay)
            param += velocity


class Nesterov(Optimizer):
    def __init__(self, model, learning_rate: float = 1e-3, momentum: float = 0.9, l2_decay: float = 1e-3) -> None:
        super().__init__(model, learning_rate, l2_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(param) for param in model.params]

    def update(self, dparams) -> None:
        for param, dparam, velocity in zip(self.model.params, dparams, self.velocities):
            velocity *= self.momentum
            velocity -= self.learning_rate * dparam
            param *= (1 - self.l2_decay)
            param += self.momentum * velocity - self.learning_rate * dparam


class RProp(Optimizer):
    def __init__(self, model, learning_rate: float = 1e-1, l2_decay: float = 1e-3,
                 positive_factor: float = 1.2,
                 negative_factor: float = 0.5,
                 min_multiplier: float = 1e-6,
                 max_multiplier: float = 50) -> None:
        super().__init__(model, learning_rate=learning_rate)
        self.positive_factor = positive_factor
        self.negative_factor = negative_factor
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.dparams_prev = tuple(np.zeros_like(param) for param in self.model.params)
        self.step_multiplers = tuple(np.ones_like(param) for param in self.model.params)

    def update(self, dparams):
        for param, dparam, dparam_prev, step_multiplier in zip(self.model.params, dparams, self.dparams_prev, self.step_multiplers):
            step_multiplier[np.sign(dparam) == np.sign(dparam_prev)] *= self.positive_factor
            step_multiplier[np.sign(dparam) != np.sign(dparam_prev)] *= self.negative_factor
            np.clip(step_multiplier, self.min_multiplier, self.max_multiplier, out=step_multiplier)
            np.copyto(dparam_prev, dparam)
            param *= (1 - self.l2_decay)
            param -= step_multiplier * self.learning_rate


class RMSProp(Optimizer):
    def __init__(self, model, learning_rate: float = 1e-2, decay: float = 0.9) -> None:
        super().__init__(model, learning_rate=learning_rate)
        self.decay = decay
        self.mean_squares = tuple(np.ones_like(param) for param in self.model.params)

    def update(self, dparams):
        for param, dparam, mean_square in zip(self.model.params, dparams, self.mean_squares):
            mean_square *= self.decay
            mean_square += (1. - self.decay) * (dparam * dparam)
            param *= (1 - self.l2_decay)
            param -= self.learning_rate * dparam / np.sqrt(mean_square)


class AdaGrad(Optimizer):
    def __init__(self, model, learning_rate: float = 1e-1, l2_decay: float = 1e-3) -> None:
        super().__init__(model, learning_rate=learning_rate)
        self.dparams_prev = tuple(np.zeros_like(param) for param in self.model.params)

    def update(self, dparams):
        for param, dparam, dparam_prev in zip(self.model.params, dparams, self.dparams_prev):
            dparam_prev += dparam * dparam
            param *= (1 - self.l2_decay)
            param -= dparam * self.learning_rate / np.sqrt(dparam_prev + 1e-8)
