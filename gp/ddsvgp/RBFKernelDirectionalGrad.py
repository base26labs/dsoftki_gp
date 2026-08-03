"""Directional-gradient RBF kernel for the DDSVGP baseline.
"""

from gpytorch.kernels.rbf_kernel import postprocess_rbf

from gp._gp_derivatives_variational_inference import load

try:
    _upstream = load("RBFKernelDirectionalGrad")
except ImportError as err:
    # The submodule has not been initialized. Stay importable -- gp/util.py
    # imports this name unconditionally -- and raise only if a DDSVGP kernel is
    # actually constructed.
    _load_error = err

    class RBFKernelDirectionalGrad:
        def __init__(self, *args, **kwargs):
            raise _load_error

else:

    class RBFKernelDirectionalGrad(_upstream.RBFKernelDirectionalGrad):
        """Upstream kernel, adapted to the GPyTorch API pinned by this project.

        GPyTorch removed ``Kernel.covar_dist``'s ``dist_postprocess_func``
        argument. The upstream ``forward`` still passes it, and because the
        modern signature absorbs unknown keywords into ``**params`` the
        postprocessing would be silently dropped -- yielding squared distances
        where the kernel expects ``exp(-d^2/2)``. Restoring the old contract
        here keeps the upstream source usable verbatim.
        """

        def covar_dist(self, x1, x2, dist_postprocess_func=None, **kwargs):
            res = super().covar_dist(x1, x2, **kwargs)
            if dist_postprocess_func is not None:
                res = dist_postprocess_func(res)
            return res


__all__ = ["RBFKernelDirectionalGrad", "postprocess_rbf"]
