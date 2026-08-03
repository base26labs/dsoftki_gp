"""Synthetic test functions that also expose their gradients.

Each ``*_with_deriv`` class returns, for a batch of inputs ``X`` of shape
``(..., d)``, a tensor of shape ``(..., 1 + d)`` whose first column is the
function value and whose remaining ``d`` columns are the gradient. This is the
format the GP models in this repository expect for derivative observations.

Gradients are obtained by reverse-mode autodifferentiation of each problem's own
``evaluate_true``, so there are no hand-maintained derivative formulas here to
drift out of sync with the function definitions.
"""

import numpy as np
import torch
from torch import Tensor

from botorch.test_functions.base import BaseTestProblem  # noqa: F401  (re-export)
from botorch.test_functions.synthetic import (
    Branin,
    Hartmann,
    SixHumpCamel,
    StyblinskiTang,
    SyntheticTestFunction,
)


class WithDeriv:
    """Mixin adding gradient observations to a BoTorch ``SyntheticTestFunction``.

    Mix in *before* the test problem so this ``evaluate_true_with_deriv`` is
    found first while ``evaluate_true`` still resolves to the problem itself::

        class Branin_with_deriv(WithDeriv, Branin): ...
    """

    def evaluate_true_with_deriv(self, X: Tensor) -> Tensor:
        x = X.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            val = self.evaluate_true(x)
            # Every entry of `val` depends only on its own row of `x`, so a
            # single backward pass over the sum recovers all per-point
            # gradients at once.
            (grad,) = torch.autograd.grad(val.sum(), x)
        val = val.detach().reshape(*X.shape[:-1], 1)
        return torch.cat([val, grad], dim=-1)

    def get_bounds(self):
        lb = np.array([lo for lo, _ in self._bounds])
        ub = np.array([hi for _, hi in self._bounds])
        return lb, ub


class Welch(SyntheticTestFunction):
    r"""Welch et al. (1992) screening function.

    Twenty-dimensional, normally evaluated on `[-0.5, 0.5]^20`:

        f(x) = 5 x_12 / (1 + x_1) + 5 (x_4 - x_20)^2 + x_5 + 40 x_19^3 - 5 x_19
               + 0.05 x_2 + 0.08 x_3 - 0.03 x_6 + 0.03 x_7 - 0.09 x_9
               - 0.01 x_10 - 0.07 x_11 + 0.25 x_13^2 - 0.04 x_14 + 0.06 x_15
               - 0.01 x_17 - 0.03 x_18

    Indices above are 1-based, matching the published statement; `x_8` and
    `x_16` do not appear. The function is a standard screening benchmark
    precisely because only a few of its twenty inputs are active.
    """

    dim = 20
    _bounds = [(-0.5, 0.5) for _ in range(dim)]
    _optimal_value = 0.0
    _optimizers = [(0.0, 0.0)]

    # Coefficients of the purely linear terms, indexed 0-based. Inputs absent
    # from the published formula (x_8, x_16, and those carrying a nonlinear
    # term) simply have coefficient zero.
    _LINEAR = {
        1: 0.05,    # x_2
        2: 0.08,    # x_3
        4: 1.0,     # x_5
        5: -0.03,   # x_6
        6: 0.03,    # x_7
        8: -0.09,   # x_9
        9: -0.01,   # x_10
        10: -0.07,  # x_11
        13: -0.04,  # x_14
        14: 0.06,   # x_15
        16: -0.01,  # x_17
        17: -0.03,  # x_18
    }

    def evaluate_true(self, X: Tensor) -> Tensor:
        linear = sum(coef * X[..., i] for i, coef in self._LINEAR.items())
        ratio = 5.0 * X[..., 11] / (1.0 + X[..., 0])
        coupling = 5.0 * (X[..., 3] - X[..., 19]) ** 2
        cubic = 40.0 * X[..., 18] ** 3 - 5.0 * X[..., 18]
        quadratic = 0.25 * X[..., 12] ** 2
        return ratio + coupling + cubic + quadratic + linear


class Branin_with_deriv(WithDeriv, Branin):
    r"""Branin, evaluated on `[-5, 10] x [0, 15]`, with gradients."""


class SixHumpCamel_with_deriv(WithDeriv, SixHumpCamel):
    r"""Six-hump camel, evaluated on `[-3, 3] x [-2, 2]`, with gradients."""


class StyblinskiTang_with_deriv(WithDeriv, StyblinskiTang):
    r"""Styblinski-Tang, evaluated on `[-5, 5]^d`, with gradients."""


class Hartmann_with_deriv(WithDeriv, Hartmann):
    r"""Hartmann (six-dimensional by default, on `[0, 1]^6`), with gradients."""


class Welch_with_deriv(WithDeriv, Welch):
    r"""Welch et al. (1992), evaluated on `[-0.5, 0.5]^20`, with gradients."""
