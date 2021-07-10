from layers import Layer, FullyConnected, RegressionID, Relu, Sigmoid, Softmax, RegressionID, Logistic
import unittest
import numpy as np
from diff_util import gradcheck, differentiate_at
from numpy.testing import assert_almost_equal
from util import softmax_loss_derivative, softmax_loss, softmax, sigmoid, sigmoid_derivative
import math
from models import Sequential


class SanityTests(unittest.TestCase):
    def test_softmax(self):
        assert_almost_equal(softmax(np.array([range(3)])), [[0.09003057, 0.24472847, 0.66524096]])

    def test_softmax_loss(self):
        os = np.array([[1., 2., 3.]])
        gradcheck(softmax_loss, (os, 1), [os], [softmax_loss_derivative(softmax(os), 1)], ['os'])

    def test_sigmoid_loss(self):
        assert_almost_equal(sigmoid_derivative(sigmoid(4)), differentiate_at(sigmoid, 4))


def test_backward(layer: Layer, x: np.ndarray, **additional_gradcheck_args):
    def forward_sum(x):
        y, _ = layer.forward(x)
        return np.sum(y)
    # We're summing all outputs, so dy is all ones
    y, cache = layer.forward(x)
    dy = np.ones_like(y)
    dparams, dx = layer.backward(x, dy, cache)
    gradcheck(forward_sum, (x,), [x, ] + layer.params, (dx,) + dparams, ['x', ] + layer.params_names, **additional_gradcheck_args)


def test_dy(loss: Layer, y: np.ndarray, targets: np.ndarray):
    dy = loss.dy(y, targets)
    assert dy.shape == y.shape == targets.shape, (dy.shape, y.shape, targets.shape)
    gradcheck(loss.loss, (y, targets), (y,), (dy,), ('y',))


def test_dx(loss: Layer, x: np.ndarray, y: np.ndarray, targets: np.ndarray):
    dx = loss.dx(y, targets)
    assert dx.shape == x.shape, (dx.shape, x.shape)

    def loss_from_x(x):
        y, cache = loss.forward(x)
        return loss.loss(y, targets)
    gradcheck(loss_from_x, (x,), (x,), (dx,), ('x',))


class TestBackwardMethods(unittest.TestCase):
    def test_fully_connected_backward_sizes(self):
        layer=FullyConnected(5, 3)
        x=np.random.uniform(0, 1, (5, 1))
        y, cache = layer.forward(x)
        dparams, dx = layer.backward(x, np.ones_like(y), cache)
        dW, db = dparams
        self.assertEqual(dW.shape, layer.W.shape)
        self.assertEqual(db.shape, layer.b.shape)
        self.assertEqual(dx.shape, x.shape)

    def test_fully_connected_backward(self):
        test_backward(layer=FullyConnected(5, 3), x=np.random.uniform(0, 1, (5, 1)))

    def test_fully_connected_backward_is_linear(self):
        # Since FC is linear, check that grad works with huge delta too.
        test_backward(layer=FullyConnected(5, 3), x=np.random.uniform(0, 1, (5, 1)), delta=100)

    def test_sigmoid_backward(self):
        test_backward(layer=Sigmoid(5), x=np.random.uniform(0, 1, (5, 1)))

    def test_relu_backward(self):
        test_backward(layer=Relu(10), x=np.random.uniform(0, 1, (5, 1)))

    def test_regression_id_backward(self):
        test_backward(layer=RegressionID(4), x=np.random.uniform(0, 1, (4, 1)))

    def test_regression_id_backward_is_linear(self):
        # Since ID is linear, check that grad works with huge delta too.
        test_backward(layer=FullyConnected(5, 3), x=np.random.uniform(0, 1, (5, 1)), delta=100)


class TestLosses(unittest.TestCase):
    def test_logistic_dy(self):
        test_dy(loss=Logistic(), y=np.array([[0.5]]).T, targets=np.array([[0]]))
        test_dy(loss=Logistic(), y=np.array([[0.5]]).T, targets=np.array([[1]]))
        test_dy(loss=Logistic(), y=np.array([[1e-1]]).T, targets=np.array([[1]]))
        test_dy(loss=Logistic(), y=np.array([[1. - 1e-1]]).T, targets=np.array([[1]]))

    def test_logistic_dx(self):
        layer = Logistic()
        x = np.random.uniform(-5, 5, (1, 1))
        y, cache = layer.forward(x)
        test_dx(loss=Logistic(), x=x, y=y, targets=np.random.choice(2, 1))

    def test_logistic_loss(self):
        self.assertAlmostEqual(Logistic.loss(np.array([[0.]]), np.array([[0]])), 0)
        self.assertAlmostEqual(Logistic.loss(np.array([[0.]]), np.array([[1]])), math.inf)
        self.assertAlmostEqual(Logistic.loss(np.array([[1.]]), np.array([[0]])), math.inf)
        self.assertAlmostEqual(Logistic.loss(np.array([[1.]]), np.array([[1]])), 0)

    def test_regression_dy(self):
        test_dy(loss=RegressionID(3), y=np.random.uniform(-5, 5, (3, 1)), targets=np.random.uniform(-10, 10, (3, 1)))

    def test_regression_dx(self):
        x = y = np.random.uniform(-5, 5, (3, 1))
        test_dx(loss=RegressionID(3), x=x, y=y, targets=np.random.uniform(-10, 10, (3, 1)))

    def test_softmax_dx(self):
        layer = Softmax(10)
        x = np.random.uniform(-5, 5, (10, 1))
        y, cache = layer.forward(x)
        targets = np.random.choice(10, 1)
        test_dx(layer, x, y=y, targets=targets)


class TestSequential(unittest.TestCase):
    def test_sequential_sizes(self):
        model = Sequential()
        model.addLayer(FullyConnected(3, 5))
        model.addLayer(FullyConnected(5, 4))
        model.addLayer(Softmax(4))
        x = np.random.uniform(0, 100, (3, 1))
        activations, caches = model.forward(x)
        dparams_for_all_layers = model.backward(activations, caches, np.array([[2]]))
        assert len(dparams_for_all_layers) == len(model.layers)
        for dparams, layer in zip(dparams_for_all_layers, model.layers):
            if layer == model.loss_layer:
                assert dparams is None
                continue
            assert len(dparams) == len(layer.params)
            for dparam, param in zip(dparams, layer.params):
                assert dparam.shape == param.shape


if __name__ == '__main__':
    unittest.main()
