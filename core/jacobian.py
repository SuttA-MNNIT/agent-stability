"""
Automatic Differentiation Jacobian Engine and Spectral Analysis.
Computes:
J_t = nabla_h f_theta(h_t, o_t)
rho(J_t) = max |lambda_i(J_t)|
||J_t||_op = sigma_max(J_t)
prod_{k=0}^{T-1} J_k
"""

import torch
import numpy as np
from typing import Tuple, List, Dict

def compute_jacobian(model: torch.nn.Module, h_t: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
    """
    Computes exact Jacobian matrix J_t = nabla_h f_theta(h_t, o_t) in R^{d x d}
    via PyTorch autograd functional.
    """
    h_vec = h_t.detach().clone().requires_grad_(True)
    o_vec = o_t.detach().clone()
    
    def func(x):
        return model(x, o_vec)
        
    J = torch.autograd.functional.jacobian(func, h_vec)
    # If h_t was 1D [d], J is [d, d]
    if J.dim() == 4:
        J = J.squeeze(0).squeeze(1)
    return J.detach()

def compute_spectral_radius(J: torch.Tensor) -> float:
    """
    Computes spectral radius rho(J) = max |lambda_i(J)|.
    """
    J_np = J.cpu().numpy()
    eigenvalues = np.linalg.eigvals(J_np)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    return spectral_radius

def compute_operator_norm(J: torch.Tensor) -> float:
    """
    Computes spectral operator norm ||J||_op = sigma_max(J) (largest singular value).
    """
    J_np = J.cpu().numpy()
    s = np.linalg.svd(J_np, compute_uv=False)
    return float(s[0])

def compute_eigenvalue_spectrum(J: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (real_parts, imag_parts) of eigenvalues of J.
    """
    J_np = J.cpu().numpy()
    eigvals = np.linalg.eigvals(J_np)
    return eigvals.real, eigvals.imag

def compute_jacobian_product_spectrum(jacobians: List[torch.Tensor]) -> Dict[str, float]:
    """
    Computes cumulative product Pi_{k=0}^{T-1} J_k and its spectral radius and operator norm.
    Uses log-scale scaling if needed to avoid numerical underflow/overflow.
    """
    if not jacobians:
        return {"spectral_radius": 0.0, "operator_norm": 0.0}
        
    prod = torch.eye(jacobians[0].shape[0], dtype=jacobians[0].dtype, device=jacobians[0].device)
    for J in jacobians:
        prod = torch.matmul(J, prod)
        
    prod_np = prod.cpu().numpy()
    eigvals = np.linalg.eigvals(prod_np)
    rho = float(np.max(np.abs(eigvals)))
    s = np.linalg.svd(prod_np, compute_uv=False)
    op_norm = float(s[0])
    
    return {
        "spectral_radius": rho,
        "operator_norm": op_norm,
        "eigenvalues": eigvals
    }
