"""DDSVGP: a variational GP whose inducing outputs include directional derivatives."""

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution

from gp.ddsvgp.DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy


class DDSVGP(ApproximateGP):
    """Sparse variational GP over function values and directional derivatives.

    Rather than carrying a full gradient at every inducing point, each of the
    ``m`` points carries derivatives along ``p`` chosen directions, giving
    ``m * (1 + p)`` variational parameters. ``inducing_directions`` holds the
    directions for all points in a single flat block, so ``p`` is recovered as
    the ratio of its length to the number of inducing points.

    Args:
        inducing_points: ``(m, d)`` tensor of initial inducing locations.
        inducing_directions: ``(m * p, d)`` tensor of directions, blocked by point.
        kernel: base kernel producing the directional-derivative block covariance
            (e.g. :class:`RBFKernelDirectionalGrad`).
        use_scale: wrap ``kernel`` in a learned ``ScaleKernel``.
        learn_inducing_locations: optimize the inducing locations during training.
    """

    def __init__(
        self,
        inducing_points,
        inducing_directions,
        kernel,
        use_scale=True,
        learn_inducing_locations=True,
    ):
        # Set before super().__init__() because the variational strategy is
        # constructed with `self` and reads these during its own setup.
        self.num_inducing = len(inducing_points)
        self.num_directions = len(inducing_directions) // self.num_inducing
        num_variational = self.num_inducing * (1 + self.num_directions)

        super().__init__(
            DirectionalGradVariationalStrategy(
                self,
                inducing_points,
                inducing_directions,
                CholeskyVariationalDistribution(num_variational),
                learn_inducing_locations=learn_inducing_locations,
            )
        )

        self.mean_module = ConstantMean()
        self.use_scale = use_scale
        self.covar_module = ScaleKernel(kernel) if use_scale else kernel

    def forward(self, x, **params):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x, **params))

    def get_lengthscale(self) -> float:
        kernel = self.covar_module.base_kernel if self.use_scale else self.covar_module
        return kernel.lengthscale.cpu()

    def get_outputscale(self) -> float:
        return self.covar_module.outputscale.cpu() if self.use_scale else 1.0
