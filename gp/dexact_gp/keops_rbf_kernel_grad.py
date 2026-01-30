"""
KeOps implementation of RBFKernelGrad for efficient gradient kernel computations.
"""

import torch
from pykeops.torch import LazyTensor
from gpytorch.kernels import Kernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.constraints import Positive
from gpytorch.priors import Prior
from gpytorch.lazy import LazyTensor as GPyTorchLazyTensor
from typing import Optional


class KeOpsRBFKernelGrad(RBFKernel):
    r"""
    Implements the RBF kernel with gradient support using KeOps for efficient computation.
    
    This kernel computes the covariance between function values and their derivatives.
    For inputs x1 and x2, it computes:
    - K(x1, x2): standard RBF kernel between function values
    - ∂K/∂x1: gradient of kernel w.r.t. first argument  
    - ∂K/∂x2: gradient of kernel w.r.t. second argument
    - ∂²K/∂x1∂x2: second derivatives (Hessian block)
    
    The output is a block matrix:
    [K(x1,x2)     ∂K/∂x2]
    [∂K/∂x1    ∂²K/∂x1∂x2]
    
    This KeOps version provides memory-efficient computation for large datasets.
    
    Args:
        ard_num_dims: Set this if you want a separate lengthscale for each input
            dimension. It should be `d` if x1 is a `n x d` matrix.
        batch_shape: Set this if you want a separate lengthscale for each batch.
        active_dims: Compute covariance for only specified input dimensions.
        lengthscale_prior: Prior for the lengthscale parameter.
        lengthscale_constraint: Constraint for the lengthscale parameter.
        eps: Minimum value for lengthscale to prevent divide by zero.
    """
    
    def __init__(
        self,
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[tuple] = None,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Positive] = None,
        eps: float = 1e-6,
        **kwargs
    ):
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            eps=eps,
            **kwargs
        )
    
    def forward(self, x1, x2, diag=False, **params):
        """
        Compute the kernel matrix with gradients using KeOps.
        
        Args:
            x1: First input tensor (n1 x d)
            x2: Second input tensor (n2 x d)
            diag: If True, only compute diagonal elements
            
        Returns:
            Kernel matrix of shape ((n1*(d+1)) x (n2*(d+1)))
        """
        if diag:
            return self._diag_forward(x1, x2, **params)
        
        batch_shape = x1.shape[:-2]
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]
        
        # Scale inputs by lengthscale
        x1_ = x1 / self.lengthscale
        x2_ = x2 / self.lengthscale
        
        # Convert to KeOps LazyTensors
        # x1_lazy[i,j] = x1_[i]
        # x2_lazy[i,j] = x2_[j]
        x1_lazy = LazyTensor(x1_[:, None, :])  # (n1, 1, d)
        x2_lazy = LazyTensor(x2_[None, :, :])  # (1, n2, d)
        
        # Compute squared distances
        # D_ij = ||x1_i - x2_j||^2
        D2 = ((x1_lazy - x2_lazy) ** 2).sum(-1)
        
        # RBF kernel: K_ij = exp(-0.5 * D_ij^2)
        K_lazy = (-0.5 * D2).exp()
        
        # Compute differences for gradient blocks
        diff = x1_lazy - x2_lazy  # (n1, n2, d)
        
        # Initialize output tensor
        K = torch.zeros(*batch_shape, n1 * (d + 1), n2 * (d + 1), 
                       device=x1.device, dtype=x1.dtype)
        
        # Fill kernel block K(x1, x2)
        K[..., :n1, :n2] = K_lazy.sum(dim=1).view(n1, n2)
        
        # Compute gradient blocks
        # ∂K/∂x2 block (n1 x (n2*d))
        for j in range(d):
            # ∂K_ij/∂x2_j = K_ij * (x1_i - x2_j) / lengthscale^2
            grad_x2_j = (K_lazy * diff[:, :, j] / self.lengthscale[j]).sum(dim=1).view(n1, n2)
            K[..., :n1, n2*(j+1):n2*(j+2)] = grad_x2_j
        
        # ∂K/∂x1 block ((n1*d) x n2)  
        for i in range(d):
            # ∂K_ij/∂x1_i = -K_ij * (x1_i - x2_j) / lengthscale^2
            grad_x1_i = (-K_lazy * diff[:, :, i] / self.lengthscale[i]).sum(dim=1).view(n1, n2)
            K[..., n1*(i+1):n1*(i+2), :n2] = grad_x1_i
        
        # Hessian block ∂²K/∂x1∂x2 ((n1*d) x (n2*d))
        for i in range(d):
            for j in range(d):
                if i == j:
                    # Diagonal terms: ∂²K/∂x1_i∂x2_i
                    hess_ij = K_lazy * (1.0/self.lengthscale[i]**2 - 
                                       (diff[:, :, i]**2)/self.lengthscale[i]**4)
                else:
                    # Off-diagonal terms: ∂²K/∂x1_i∂x2_j  
                    hess_ij = K_lazy * (diff[:, :, i] * diff[:, :, j]) / (
                        self.lengthscale[i]**2 * self.lengthscale[j]**2)
                
                K[..., n1*(i+1):n1*(i+2), n2*(j+1):n2*(j+2)] = hess_ij.sum(dim=1).view(n1, n2)
        
        # Apply permutation to match GPyTorch's MultiTask ordering
        # Interleave function values and derivatives for each point
        pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().reshape((n1 * (d + 1)))
        pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().reshape((n2 * (d + 1)))
        K = K[..., pi1, :][..., :, pi2]
        
        return K
    
    def _diag_forward(self, x1, x2, **params):
        """
        Compute only diagonal elements of the kernel matrix with gradients.
        """
        if not (x1.shape == x2.shape and torch.allclose(x1, x2)):
            raise RuntimeError("diag=True only works when x1 == x2")
        
        batch_shape = x1.shape[:-2]
        n, d = x1.shape[-2:]
        
        # Diagonal of kernel block is all ones (after RBF transformation)
        kernel_diag = torch.ones(*batch_shape, n, device=x1.device, dtype=x1.dtype)
        
        # Diagonal of gradient blocks is 1/lengthscale^2 for each dimension
        grad_diag = torch.ones(*batch_shape, n, d, device=x1.device, dtype=x1.dtype)
        grad_diag = grad_diag / self.lengthscale.pow(2)
        grad_diag = grad_diag.reshape(*batch_shape, n * d)
        
        # Combine and permute
        k_diag = torch.cat([kernel_diag, grad_diag], dim=-1)
        pi = torch.arange(n * (d + 1)).view(d + 1, n).t().reshape((n * (d + 1)))
        
        return k_diag[..., pi]
    
    def num_outputs_per_input(self, x1, x2):
        """Returns the number of outputs per input (1 function value + d derivatives)."""
        return x1.shape[-1] + 1


# Example usage and testing
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test dimensions
    n1, n2 = 100, 80
    d = 3
    
    # Generate test data
    x1 = torch.randn(n1, d)
    x2 = torch.randn(n2, d)
    
    # Create kernel
    kernel = KeOpsRBFKernelGrad(ard_num_dims=d)
    
    # Compute kernel matrix
    K = kernel(x1, x2)
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Expected shape: ({n1*(d+1)}, {n2*(d+1)})")
    
    # Test diagonal computation
    K_diag = kernel(x1, x1, diag=True)
    print(f"Diagonal shape: {K_diag.shape}")
    print(f"Expected diagonal shape: ({n1*(d+1)},)")
    
    # Verify positive definiteness of K(x1, x1)
    K_self = kernel(x1, x1)
    try:
        L = torch.linalg.cholesky(K_self + 1e-6 * torch.eye(K_self.shape[0]))
        print("Kernel matrix is positive definite ✓")
    except:
        print("Warning: Kernel matrix may not be positive definite")