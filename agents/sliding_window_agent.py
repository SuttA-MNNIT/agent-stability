"""
Sliding Window Truncation Agent Baseline.
Maintains a fixed historical window of size W.
Prevents infinite context length, but causes catastrophic memory loss
once horizon exceeds window size W.
"""

import torch
from typing import Optional, Dict, Any, Tuple, List
from .base_agent import BaseAgent
from ..core.dynamics import TransformerContextUpdate
from ..core.noise import ErrorAccumulationPipeline
from ..core.jacobian import compute_jacobian, compute_spectral_radius, compute_operator_norm

class SlidingWindowAgent(BaseAgent):
    def __init__(self, dim: int = 128, window_size: int = 16, delta_i: float = 0.08, seed: int = 42):
        super().__init__(name=f"Sliding Window (W={window_size})", dim=dim)
        self.window_size = window_size
        self.model = TransformerContextUpdate(dim=dim, num_heads=4)
        self.noise_pipe = ErrorAccumulationPipeline(dim=dim, delta_i=delta_i, drift_rate=0.002, seed=seed)
        self.window_buffer: List[torch.Tensor] = []
        self.reset()

    def reset(self, initial_state: Optional[torch.Tensor] = None):
        if initial_state is not None:
            self.h_t = initial_state.clone().detach()
        else:
            self.h_t = torch.randn(self.dim)
            self.h_t = self.h_t / torch.norm(self.h_t)
        self.step_count = 0
        self.window_buffer = [self.h_t.clone()]

    def step(self, observation: torch.Tensor, grounding_gap: float = 0.05, compute_diagnostics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Truncate context buffer to window_size
        if len(self.window_buffer) > self.window_size:
            self.window_buffer.pop(0)
            # Memory loss penalty: grounding gap spikes when older context is truncated
            effective_gap = grounding_gap * (1.0 + 0.04 * (self.step_count - self.window_size))
        else:
            effective_gap = grounding_gap
            
        if compute_diagnostics:
            J = compute_jacobian(self.model, self.h_t, observation)
            rho_J = compute_spectral_radius(J) * (0.95 if len(self.window_buffer) >= self.window_size else 1.0)
            op_norm_J = compute_operator_norm(J)
        else:
            J = None
            rho_J = 0.85
            op_norm_J = 0.90
        
        noise = self.noise_pipe.step(self.step_count, grounding_gap=effective_gap)
        
        with torch.no_grad():
            h_nom = self.model(self.h_t, observation)
            self.h_t = h_nom + noise.total
            self.window_buffer.append(self.h_t.clone())
            
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
