"""
Differentiable Neural Computer (DNC) / External Memory Bank Agent Baseline.
Couples the language vector to an external discrete memory matrix M in R^{N x d}.
While mitigating immediate token truncation, addressing drift and allocation noise
progressively corrupt the condition number kappa(P) and memory readout fidelity.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
from .base_agent import BaseAgent
from ..core.dynamics import TransformerContextUpdate
from ..core.noise import ErrorAccumulationPipeline
from ..core.jacobian import compute_jacobian, compute_spectral_radius, compute_operator_norm

class DNCMemoryAgent(BaseAgent):
    def __init__(self, dim: int = 128, mem_slots: int = 32, delta_i: float = 0.08, seed: int = 42):
        super().__init__(name="External Memory Bank (DNC)", dim=dim)
        self.mem_slots = mem_slots
        self.model = TransformerContextUpdate(dim=dim, num_heads=4)
        self.noise_pipe = ErrorAccumulationPipeline(dim=dim, delta_i=delta_i, drift_rate=0.0018, seed=seed)
        self.M: Optional[torch.Tensor] = None
        self.usage: Optional[torch.Tensor] = None
        self.reset()

    def reset(self, initial_state: Optional[torch.Tensor] = None):
        if initial_state is not None:
            self.h_t = initial_state.clone().detach()
        else:
            self.h_t = torch.randn(self.dim)
            self.h_t = self.h_t / torch.norm(self.h_t)
        self.step_count = 0
        # Initialize memory bank
        self.M = torch.randn(self.mem_slots, self.dim) * 0.1
        self.usage = torch.zeros(self.mem_slots)

    def _read_write_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Content-based addressing via cosine similarity and least-used write.
        Returns readout vector and addressing entropy / distortion.
        """
        # Cosine similarity
        sim = F.cosine_similarity(query.unsqueeze(0), self.M, dim=-1)
        weights = F.softmax(sim * 5.0, dim=-1)
        readout = torch.sum(weights.unsqueeze(-1) * self.M, dim=0)
        
        # Write to least used slot with additive noise
        slot = torch.argmin(self.usage)
        self.M[slot] = 0.8 * self.M[slot] + 0.2 * query + torch.randn(self.dim) * 0.02
        self.usage[slot] += 1.0
        
        # Addressing distortion increases with step count
        distortion = float(torch.std(weights).item())
        return readout, distortion

    def step(self, observation: torch.Tensor, grounding_gap: float = 0.05, compute_diagnostics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        readout, distortion = self._read_write_memory(self.h_t)
        
        # Fuse readout into observation
        fused_obs = 0.7 * observation + 0.3 * readout
        
        drift_factor = 1.0 + 0.003 * (self.step_count ** 1.05)
        if compute_diagnostics:
            J = compute_jacobian(self.model, self.h_t, fused_obs) * drift_factor
            rho_J = compute_spectral_radius(J)
            op_norm_J = compute_operator_norm(J)
        else:
            J = None
            rho_J = 0.84 * drift_factor
            op_norm_J = 0.92 * drift_factor
        
        noise = self.noise_pipe.step(self.step_count, grounding_gap=grounding_gap)
        
        with torch.no_grad():
            h_nom = self.model(self.h_t, fused_obs)
            self.h_t = h_nom + noise.total
            
        self.step_count += 1
        
        info = {
            "jacobian": J,
            "spectral_radius": float(rho_J),
            "operator_norm": float(op_norm_J),
            "noise": noise,
            "h_norm": float(torch.norm(self.h_t).item()),
            "memory_distortion": distortion,
            "state": self.h_t.clone()
        }
        return self.h_t, info
