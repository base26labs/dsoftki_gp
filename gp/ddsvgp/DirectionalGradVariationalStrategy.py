"""Directional-gradient variational strategy for the DDSVGP baseline.
"""

from gpytorch import settings
from gpytorch.lazy import TriangularLazyTensor, delazify
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.memoize import cached

from gp._gp_derivatives_variational_inference import load

try:
    _upstream = load("DirectionalGradVariationalStrategy")
except ImportError as err:
    # The submodule has not been initialized. Stay importable so run.py can be
    # started for any other model, and raise only if DDSVGP is actually used.
    _load_error = err

    class DirectionalGradVariationalStrategy:
        def __init__(self, *args, **kwargs):
            raise _load_error

else:

    class DirectionalGradVariationalStrategy(_upstream.DirectionalGradVariationalStrategy):
        """Upstream strategy, adapted to the GPyTorch API pinned by this project.

        GPyTorch made ``settings.cholesky_jitter.value()`` require a ``dtype``;
        the upstream ``_cholesky_factor`` calls it with no arguments and raises
        a ``TypeError`` under the pinned version. This override reproduces
        GPyTorch's own cholesky-factor caching while selecting the jitter for
        the dtype the factorization actually runs in (``float64``).
        """

        @cached(name="cholesky_factor", ignore_args=True)
        def _cholesky_factor(self, induc_induc_covar):
            covar = delazify(induc_induc_covar).double()
            L = psd_safe_cholesky(covar, jitter=settings.cholesky_jitter.value(covar.dtype))
            return TriangularLazyTensor(L)


__all__ = ["DirectionalGradVariationalStrategy"]
