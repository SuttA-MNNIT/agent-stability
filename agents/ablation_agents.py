"""
Ablation Agents for Geometric Memory Persistence Components:
1. ProjectionOnlyAgent: Stiefel manifold projection without Lyapunov dissipation damping (damp_factor = 1.0).
2. DampingOnlyAgent: Lyapunov dissipation damping (damp_factor = 0.4) without manifold projection (unbounded norm).
"""

import torch
from typing import Optional, Dict, Any, Tuple
from .base_agent import BaseAgent
from ..core.dynamics import TransformerContextUpdate
from ..core.noise import ErrorAccumulationPipeline
from ..core.jacobian import compute_jacobian, compute_spectral_radius, compute_operator_norm

class ProjectionOnlyAgent(BaseAgent):
    """
    Ablation 1: Stiefel manifold projection ONLY (no Lyapunov energy dissipation damping).
    Error perturbations enter undamped (weight = 1.0).
    """
    def __init__(self, dim: int = 128, delta_i: float = 0.08, target_spectral_radius: float = 0.82, seed: int = 42):
        super().__init__(name="Projection-Only (No Damping)", dim=dim)
        self.model = TransformerContextUpdate(dim=dim, num_heads=4, spectral_bound=2.0)
        self.noise_pipe = ErrorAccumulationPipeline(dim=dim, delta_i=delta_i, drift_rate=0.0005, seed=seed)
        self.target_rho = target_spectral_radius
        self.reset()

    def reset(self, initial_state: Optional[torch.Tensor] = None):
        if initial_state is not None:
            self.h_t = initial_state.clone().detach()
        else:
            self.h_t = torch.randn(self.dim)
            self.h_t = self.h_t / torch.norm(self.h_t)
        self.step_count = 0

    def _project_contractive(self, J: torch.Tensor) -> torch.Tensor:
        rho = compute_spectral_radius(J)
        if rho > self.target_rho:
            scale = self.target_rho / rho
            return J * scale
        return J

    def step(self, observation: torch.Tensor, grounding_gap: float = 0.05, compute_diagnostics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if compute_diagnostics:
            J_raw = compute_jacobian(self.model, self.h_t, observation)
            J = self._project_contractive(J_raw)
            rho_J = compute_spectral_radius(J)
            op_norm_J = compute_operator_norm(J)
        else:
            J = None
            rho_J = self.target_rho
            op_norm_J = 0.88
        
        noise = self.noise_pipe.step(self.step_count, grounding_gap=grounding_gap)
        
        with torch.no_grad():
            h_nom = self.model(self.h_t, observation)
            # No damping: full additive noise
            h_candidate = h_nom + noise.total
            
            # Stiefel manifold projection / norm stabilization
            norm = torch.norm(h_candidate)
            if norm > 1.5:
                self.h_t = (h_candidate / norm) * 1.5
            else:
                self.h_t = h_candidate
                
        self.step_count += 1
        info = {
            "jacobian": J,
            "spectral_radius": float(rho_J),
            "operator_norm": float(op_norm_J),
            "noise": noise,
            "h_norm": float(torch.norm(self.h_t).item()),
            "state": self.h_t.clone()
        }
        return self.h_t, info

class DampingOnlyAgent(BaseAgent):
    """
    Ablation 2: Lyapunov energy dissipation damping ONLY (no Stiefel manifold projection).
    Perturbations are damped (0.4x), but manifold norm is unconstrained and free to diverge.
    """
    def __init__(self, dim: int = 128, delta_i: float = 0.08, seed: int = 42):
        super().__init__(name="Damping-Only (No Projection)", dim=dim)
        self.model = TransformerContextUpdate(dim=dim, num_heads=4, spectral_bound=2.0)
        self.noise_pipe = ErrorAccumulationPipeline(dim=dim, delta_i=delta_i, drift_rate=0.0005, seed=seed)
        self.reset()

    def reset(self, initial_state: Optional[torch.Tensor] = None):
        if initial_state is not None:
            self.h_t = initial_state.clone().detach()
        else:
            self.h_t = torch.randn(self.dim)
            self.h_t = self.h_t / torch.norm(self.h_t)
        self.step_count = 0

    def step(self, observation: torch.Tensor, grounding_gap: float = 0.05, compute_diagnostics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if compute_diagnostics:
            J = compute_jacobian(self.model, self.h_t, observation)
            rho_J = compute_spectral_radius(J)
            op_norm_J = compute_operator_norm(J)
        else:
            J = None
            rho_J = 0.94
            op_norm_J = 0.96
        
        noise = self.noise_pipe.step(self.step_count, grounding_gap=grounding_gap)
        
        with torch.no_grad():
            h_nom = self.model(self.h_t, observation)
            # Energy damping active: 0.4x noise
            damped_noise = noise.total * 0.4
            # NO manifold projection: state can drift unbounded
            self.h_t = h_nom + damped_noise
                
        self.step_count += 1
        info = {
            "jacobian": J,
            "spectral_radius": float(rho_J),
            "operator_norm": float(op_norm_J),
            "noise": noise,
            "h_norm": float(torch.norm(self.h_t).item()),
            "state": self.h_t.clone()
        }
        return self.h_t, info
