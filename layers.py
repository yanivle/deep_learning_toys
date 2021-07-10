import numpy as np
from util import sigmoid, sigmoid_derivative
from util import sigmoid_derivative, softmax, softmax_loss_derivative


class Layer:
    def __init__(self, input_size: int, output_size: int, name: str = None):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name if name else type(self).__name__
        self.params = []
        self.params_names = []

    def addParam(self, name: str, param: np.ndarray) -> None:
        self.params.append(param)
        self.params_names.append(name)
        setattr(self, name, param)

    def __repr__(self) -> str:
        params_str = '\n  '.join([f'{name}({param.size} parameters)' for param, name in zip(self.params, self.params_names)])
        return f'Layer: {self.name}({self.num_params} parameters):\n  {params_str}'

    def details(self) -> str:
        params_str = '\n  '.join([f'{name}({param.size} parameters): {param}' for param, name in zip(self.params, self.params_names)])
        return f'Layer: {self.name}({self.num_params} parameters):\n  {params_str}'

    @property
    def num_params(self) -> int:
        return sum(param.size for param in self.params)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        '''Returns y, cache.'''
        raise NotImplementedError()

    def backward(self, x: np.ndarray, dy: np.ndarray, cache: dict) -> tuple[list[np.ndarray], np.ndarray]:
        '''Returns dparams, dx.'''
        raise NotImplementedError()

    # The below methods only exist for loss layers.
    @staticmethod
    def loss(y: np.ndarray, targets: np.ndarray) -> float:
        '''The scalar loss.'''
        raise NotImplementedError()

    @staticmethod
    def dy(y: np.ndarray, targets: np.ndarray) -> np.ndarray:
        '''dloss/dy
        This method is only used for testing (models use dx instead).'''
        raise NotImplementedError()

    @staticmethod
    def dx(y: np.ndarray, targets: np.ndarray) -> np.ndarray:
        '''dloss/dx'''
        raise NotImplementedError()


class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int, name: str = None, epsilon_init: float = 1e-5, bias_pref: float = 0):
        '''A fully connected linear layer.

        bias_pref can be set to a positive number (e.g. 0.1) when this is coupled with a Relu unit, to prevent 0 initial derivative.'''
        super().__init__(input_size, output_size, name=name)
        self.addParam('W', np.random.randn(output_size, input_size) * epsilon_init)
        self.addParam('b', np.zeros((self.output_size, 1)) + bias_pref)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        return np.dot(self.W, x) + self.b, {}

    def backward(self, x: np.ndarray, dy: np.ndarray, cache: dict) -> tuple[list[np.ndarray], np.ndarray]:
        '''Returns dparams, dx.'''
        dW = np.dot(dy, x.T)
        db = np.copy(dy)
        dx = np.dot(self.W.T, dy)
        assert dW.shape == self.W.shape, (dW.shape, self.W.shape)
        assert db.shape == self.b.shape, (db.shape, self.b.shape)
        assert dx.shape == x.shape, (dx.shape, x.shape)
        return (dW, db), dx


class Relu(Layer):
    def __init__(self, size: int, name: str = None):
        super().__init__(size, size, name=name)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        y = np.copy(x)
        y[x < 0] = 0
        return y, {}

    def backward(self, x: np.ndarray, dy: np.ndarray, cache: dict) -> tuple[list[np.ndarray], np.ndarray]:
        dx = np.copy(dy)
        dx[x < 0] = 0
        dparams = tuple()
        return dparams, dx


class Sigmoid(Layer):
    def __init__(self, size: int, name: str = None):
        super().__init__(size, size, name=name)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        y = sigmoid(x)
        return y, {'y': y}

    def backward(self, x: np.ndarray, dy: np.ndarray, cache: dict) -> tuple[list[np.ndarray], np.ndarray]:
        dparams = tuple()
        dx = sigmoid_derivative(cache['y']) * dy
        return dparams, dx


class Logistic(Sigmoid):
    def __init__(self, name: str=None):
        super().__init__(1, name=name)

    @staticmethod
    def loss(y: np.ndarray, targets: np.ndarray) -> float:
        assert targets.size == 1, targets.size
        assert targets.item() in [0, 1]
        assert y.size == 1, y
        target = targets.item()
        p = y.item()
        return -np.log(p) if target == 1 else -np.log(1 - p)

    @staticmethod
    def dy(y: np.ndarray, targets: np.ndarray) -> np.ndarray:
        assert targets.size == 1, targets.size
        target = targets.item()
        assert target in [0, 1], target
        assert y.size == 1, y
        if target == 0:
            return 1 / (1 - y)
        else:
            return -1 / y

    @staticmethod
    def dx(y: np.ndarray, targets: np.ndarray) -> np.ndarray:
        assert targets.size == 1, targets.size
        target = targets.item()
        assert target in [0, 1], target
        if target == 0:
            return y
        else:
            return y - 1.


class Softmax(Layer):
    def __init__(self, size: int, name: str = None):
        super().__init__(size, size, name=name)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        y = softmax(x)
        return y, {'y': y}

    # Not implementing backward (which would require a matrix of derivatives and I'm too tired to think about it :))
    # So softmax can only be used as a loss layer.
    # Also, not implementing dy :)

    @staticmethod
    def loss(y: np.ndarray, targets: np.ndarray) -> float:
        assert targets.size == 1, targets.size
        assert np.issubdtype(targets.dtype, np.integer)
        return -np.log(y[targets.item()])

    @staticmethod
    def dx(y: np.ndarray, targets: np.ndarray) -> np.ndarray:
        assert targets.size == 1, targets.size
        assert np.issubdtype(targets.dtype, np.integer)
        target = targets.item()
        dx = np.copy(y)
        dx[target, 0] -= 1
        return dx


class RegressionID(Layer):
    '''This is a layer that does nothing (ID) except for allowing to propagate the loss backwards.'''
    def __init__(self, size: int, name: str = None):
        super().__init__(size, size, name=name)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        # Probably don't have to copy here, but too lazy to make sure right now.
        return np.copy(x), {}

    def backward(self, x: np.ndarray, dy: np.ndarray, cache: dict) -> tuple[list[np.ndarray], np.ndarray]:
        dparams = tuple()
        dx = np.ones_like(x)
        return dparams, dx

    @staticmethod
    def loss(y: np.ndarray, targets: np.ndarray) -> float:
        d = y - targets
        return np.sum(d * d)

    @staticmethod
    def dy(y: np.ndarray, targets: np.ndarray) -> np.ndarray:
        assert y.shape == targets.shape
        return 2 * (y - targets)

    @staticmethod
    def dx(y: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return RegressionID.dy(y, targets)
