import numpy as np
from numpy.testing import assert_almost_equal


def differentiate_at(f, x: float, delta: float = 1e-5) -> float:
    return (f(x + delta) - f(x - delta)) / (2 * delta)


assert_almost_equal(differentiate_at(lambda x: x**2, 4), 8)


def differentiate_multivariate(f, args, params, delta: float = 1e-5):
    dparams = [np.zeros_like(param) for param in params]
    for dparam, param in zip(dparams, params):
        for i in range(param.size):
            org = param.flat[i]
            param.flat[i] = org + delta
            r1 = f(*args)
            param.flat[i] = org - delta
            r2 = f(*args)
            param.flat[i] = org
            dparam.flat[i] = (r1 - r2) / (2. * delta)
    return dparams


def gradcheck(f, args, params, dparams, params_names, delta: float = 1e-5, epsilon: float = 1e-7):
    dparams_approx = differentiate_multivariate(f, args, params, delta)
    for dparam, dparam_approx, param_name in zip(dparams, dparams_approx, params_names):
        rel_err = abs(dparam_approx - dparam) / \
            abs(dparam_approx + dparam + np.spacing(1))
        assert (rel_err < epsilon).all(), (param_name,
                                           'dparam', dparam, 'dparam_approx', dparam_approx, 'rel_err', rel_err)
