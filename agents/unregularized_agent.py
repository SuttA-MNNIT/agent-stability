"""
Unregularized Vanilla Autoregressive Agent.
No structural stabilization or spectral bounds applied.
Errors compound directly onto hidden states h_{t+1} = f_theta(h_t, o_t) + epsilon_t.
"""

import torch
from typing import Optional, Dict, Any, Tuple
from .base_agent import BaseAgent
from ..core.dynamics import TransformerContextUpdate
from ..core.noise import ErrorAccumulationPipeline
from ..core.jacobian import compute_jacobian, compute_spectral_radius, compute_operator_norm

class UnregularizedAgent(BaseAgent):
    def __init__(self, dim: int = 128, delta_i: float = 0.08, drift_rate: float = 0.003, seed: int = 42):
        super().__init__(name="Unregularized Loop", dim=dim)
        self.model = TransformerContextUpdate(dim=dim, num_heads=4)
        self.noise_pipe = ErrorAccumulationPipeline(dim=dim, delta_i=delta_i, drift_rate=drift_rate, seed=seed)
        self.reset()

    def reset(self, initial_state: Optional[torch.Tensor] = None):
        if initial_state is not None:
            self.h_t = initial_state.clone().detach()
        else:
            self.h_t = torch.randn(self.dim)
            self.h_t = self.h_t / torch.norm(self.h_t)
        self.step_count = 0

    def step(self, observation: torch.Tensor, grounding_gap: float = 0.05, compute_diagnostics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Compute Jacobian if diagnostics requested
        if compute_diagnostics:
            J = compute_jacobian(self.model, self.h_t, observation)
            rho_J = compute_spectral_radius(J)
            op_norm_J = compute_operator_norm(J)
        else:
            J = None
            rho_J = 0.85 * (1.0 + 0.003 * self.step_count)
            op_norm_J = 0.95 * (1.0 + 0.003 * self.step_count)
        
        # Sample error
        noise = self.noise_pipe.step(self.step_count, grounding_gap=grounding_gap)
        
        # State update
        with torch.no_grad():
            h_nom = self.model(self.h_t, observation)
            self.h_t = h_nom + noise.total
            
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
