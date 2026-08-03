"""DSVGP: a variational GP whose inducing outputs include gradients."""

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    CiqVariationalStrategy,
    NaturalVariationalDistribution,
)

from gp.dsvgp.GradVariationalStrategy import GradVariationalStrategy


class DSVGP(ApproximateGP):
    """Sparse variational GP over function values and their partial derivatives.

    Each of the ``m`` inducing points carries ``d + 1`` outputs -- the function
    value plus one partial derivative per input dimension -- so the variational
    distribution has ``m * (d + 1)`` entries rather than ``m``.

    Args:
        inducing_points: ``(m, d)`` tensor of initial inducing locations.
        kernel: base kernel; must itself produce the gradient-augmented block
            covariance (e.g. ``RBFKernelGrad``).
        use_scale: wrap ``kernel`` in a learned ``ScaleKernel``.
        variational_distribution: pass ``"NGD"`` for a natural-gradient
            parameterization; anything else uses a Cholesky factorization.
        variational_strategy: pass ``"CIQ"`` for contour-integral quadrature;
            anything else uses :class:`GradVariationalStrategy`.
    """

    def __init__(self, inducing_points, kernel, use_scale=True, **kwargs):
        num_inducing, dim = inducing_points.shape[-2], inducing_points.shape[-1]
        num_variational = num_inducing * (dim + 1)

        if kwargs.get("variational_distribution") == "NGD":
            variational_distribution = NaturalVariationalDistribution(num_variational)
        else:
            variational_distribution = CholeskyVariationalDistribution(num_variational)

        strategy = (
            CiqVariationalStrategy
            if kwargs.get("variational_strategy") == "CIQ"
            else GradVariationalStrategy
        )
        super().__init__(
            strategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )
        )

        self.mean_module = ConstantMean()
        self.use_scale = use_scale
        self.covar_module = ScaleKernel(kernel) if use_scale else kernel

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def get_lengthscale(self) -> float:
        kernel = self.covar_module.base_kernel if self.use_scale else self.covar_module
        return kernel.lengthscale.cpu()

    def get_outputscale(self) -> float:
        return self.covar_module.outputscale.cpu() if self.use_scale else 1.0
