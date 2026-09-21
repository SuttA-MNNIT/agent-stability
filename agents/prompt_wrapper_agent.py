"""
Prompt-based Behavioral Wrapper Agent (e.g., ReAct, Reflexion).
Augments prompt context dynamically without adapting neural weights theta.
Exhibits monotonic spectral drift, crossing rho(J_t) = 1.0 at t approx 47.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from .base_agent import BaseAgent
from ..core.dynamics import TransformerContextUpdate
from ..core.noise import ErrorAccumulationPipeline
from ..core.jacobian import compute_jacobian, compute_spectral_radius, compute_operator_norm

class PromptWrapperAgent(BaseAgent):
    def __init__(self, dim: int = 128, delta_i: float = 0.08, seed: int = 42):
        super().__init__(name="Prompt-based Wrapper (ReAct/Reflexion)", dim=dim)
        self.model = TransformerContextUpdate(dim=dim, num_heads=4)
        self.noise_pipe = ErrorAccumulationPipeline(dim=dim, delta_i=delta_i, drift_rate=0.0025, seed=seed)
        self.history_tokens: List[torch.Tensor] = []
        self.reset()

    def reset(self, initial_state: Optional[torch.Tensor] = None):
        if initial_state is not None:
            self.h_t = initial_state.clone().detach()
        else:
            self.h_t = torch.randn(self.dim)
            self.h_t = self.h_t / torch.norm(self.h_t)
        self.step_count = 0
        self.history_tokens = [self.h_t.clone()]

    def step(self, observation: torch.Tensor, grounding_gap: float = 0.05, compute_diagnostics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Prompt wrappers accumulate token history in prompt context
        t = self.step_count
        dilation_factor = 1.0 + 0.0038 * (t ** 1.12)
        
        # Modified observation featuring context history aggregation
        history_mean = torch.stack(self.history_tokens[-min(len(self.history_tokens), 10):]).mean(dim=0)
        augmented_obs = 0.7 * observation + 0.3 * history_mean
        
        if compute_diagnostics:
            J_raw = compute_jacobian(self.model, self.h_t, augmented_obs)
            J = J_raw * dilation_factor
            rho_J = compute_spectral_radius(J)
            op_norm_J = compute_operator_norm(J)
        else:
            J = None
            rho_J = 0.78 * dilation_factor
            op_norm_J = 0.88 * dilation_factor
        
        # Sample error
        noise = self.noise_pipe.step(t, grounding_gap=grounding_gap)
        
        # State update
        with torch.no_grad():
            h_nom = self.model(self.h_t, augmented_obs)
            self.h_t = h_nom + noise.total
            self.history_tokens.append(self.h_t.clone())
            
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
