"""
Lyapunov Energy Analysis & State-Space Geometric Stability.
Implements:
V(h_t) = h_t^T P h_t
kappa(P) = lambda_max(P) / lambda_min(P)
Delta V_t = V(h_{t+1}) - V(h_t)
Theorem 1 & Theorem 3 energy dissipation checks.
"""

import torch
import numpy as np
from typing import Tuple, Optional

class LyapunovAnalyzer:
    """
    Constructs and analyzes symmetric positive-definite energy landscapes:
    V(h) = h^T P h.
    """
    def __init__(self, dim: int = 128, condition_number: float = 1.0, seed: int = 42):
        self.dim = dim
        self.condition_number = condition_number
        self.P = self._construct_spd_matrix(dim, condition_number, seed)
        
        # Eigenvalues and conditioning
        eigvals = torch.linalg.eigvalsh(self.P)
        self.lambda_min = float(eigvals[0].item())
        self.lambda_max = float(eigvals[-1].item())
        self.kappa = self.lambda_max / max(self.lambda_min, 1e-12)

    def _construct_spd_matrix(self, dim: int, kappa: float, seed: int) -> torch.Tensor:
        """
        Constructs a random symmetric positive definite matrix P in R^{d x d}
        with precisely controlled condition number kappa.
        """
        torch.manual_seed(seed)
        # Random orthogonal matrix via QR
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        
        # Log-spaced spectrum from 1.0 to kappa
        eigenvalues = torch.logspace(0, np.log10(max(kappa, 1.0)), steps=dim)
        D = torch.diag(eigenvalues)
        
        P = Q @ D @ Q.T
        # Guarantee exact symmetry
        P = 0.5 * (P + P.T)
        return P

    def compute_energy(self, h: torch.Tensor) -> float:
        """
        Computes V(h) = h^T P h.
        Supports single vector [d] or batch [N, d].
        """
        if h.dim() == 1:
            energy = torch.matmul(h, torch.matmul(self.P, h)).item()
            return float(energy)
        else:
            # Batch computation: sum_j h_ij (P h)_ij
            Ph = torch.matmul(h, self.P.T)
            energy = torch.sum(h * Ph, dim=-1)
            return energy.cpu().numpy()

    def compute_energy_drift(self, h_curr: torch.Tensor, h_next: torch.Tensor) -> float:
        """
        Delta V_t = V(h_{t+1}) - V(h_t).
        Negative value implies dissipative/stable contraction.
        """
        v_curr = self.compute_energy(h_curr)
        v_next = self.compute_energy(h_next)
        return float(v_next - v_curr)

    def compute_theoretical_bound(self, delta_stable: float) -> float:
        """
        Evaluates RHS of Theorem 3:
        Bound = delta_stable / (sqrt(d) * kappa(P))
        """
        return delta_stable / (np.sqrt(self.dim) * self.kappa)
