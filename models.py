import numpy as np
from layers import Layer
import itertools


class Model:
    def __repr__(self) -> str:
        raise NotImplemented()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
        '''Returns activations, caches'''
        raise NotImplementedError()

    def backward(self, activations: list[np.ndarray], caches: list[dict], targets: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError()

    def loss(self, activations: list[np.ndarray], targets: np.ndarray) -> float:
        raise NotImplementedError()

    @property
    def params(self):
        raise NotImplementedError()


class Sequential(Model):
    def __init__(self) -> None:
        super().__init__()
        self.layers: list[Layer] = []

    def addLayer(self, layer: Layer):
        assert not self.layers or self.layers[-1].output_size == layer.input_size, ('Incompatible layer sizes', self.layers[-1], layer)
        self.layers.append(layer)

    def __repr__(self) -> str:
        return f'Sequential({len(self.layers)} layers):\n  ' + '\n  '.join(repr(layer) for layer in self.layers)

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
        '''Returns activations, caches'''
        activations = [x]
        caches = []
        for layer in self.layers:
            a, cache = layer.forward(x)
            activations.append(a)
            caches.append(cache)
            x = a
        return activations, caches

    def backward(self, activations: list[np.ndarray], caches: list[dict], targets: np.ndarray) -> list[np.ndarray]:
        dparams = [None] * len(self.layers)
        dy = None
        for i in reversed(range(len(self.layers))):
            layer, x, y, cache = self.layers[i], activations[i], activations[i+1], caches[i]
            if layer == self.loss_layer:
                dy = self.loss_layer.dx(y, targets)
            else:
                dparams[i], dy = layer.backward(x, dy, cache)
        return dparams

    @property
    def loss_layer(self):
        return self.layers[-1]

    def loss(self, activations: list[np.ndarray], targets: np.ndarray) -> float:
        return self.loss_layer.loss(activations[-1], targets)

    @property
    def params(self):
        return itertools.chain.from_iterable(layer.params for layer in self.layers)
