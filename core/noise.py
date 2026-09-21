"""
Error Accumulation Pipeline for Agentic Orchestration Loops.
Decomposes cumulative noise into three orthogonal components:
epsilon_t = epsilon_t^{(i)} + epsilon_t^{(f)} + epsilon_t^{(c)}
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class NoiseSample:
    total: torch.Tensor
    intrinsic: torch.Tensor
    feedback: torch.Tensor
    drift: torch.Tensor
    norm_total: float
    norm_intrinsic: float
    norm_feedback: float
    norm_drift: float

class ErrorAccumulationPipeline:
    """
    Simulates tri-partite error injection across orchestration steps:
    1. Intrinsic stochasticity: epsilon_t^{(i)} ~ N(0, sigma_i^2 I) clipped to ||epsilon^{(i)}|| <= delta_i
    2. Feedback distortion: epsilon_t^{(f)} = G * Delta_g(t)
    3. Contextual drift: epsilon_t^{(c)} representing attention loss over time
    """
    def __init__(self, dim: int = 128, delta_i: float = 0.08, drift_rate: float = 0.002, seed: Optional[int] = None):
        self.dim = dim
        self.delta_i = delta_i
        self.drift_rate = drift_rate
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def sample_intrinsic_noise(self, delta_i: Optional[float] = None) -> torch.Tensor:
        """Intrinsic stochasticity from temperature sampling and finite precision."""
        bound = delta_i if delta_i is not None else self.delta_i
        raw_noise = torch.randn(self.dim)
        raw_norm = torch.norm(raw_noise)
        if raw_norm > 0:
            scale = torch.rand(1).item() * bound
            eps_i = (raw_noise / raw_norm) * scale
        else:
            eps_i = raw_noise
        return eps_i

    def sample_feedback_distortion(self, grounding_gap: float, direction_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Feedback distortion resulting from semantic grounding gap Delta_g(t).
        ||epsilon_t^{(f)}|| is directly proportional to Delta_g(t).
        """
        if direction_vector is None:
            direction_vector = torch.randn(self.dim)
            norm = torch.norm(direction_vector)
            if norm > 0:
                direction_vector = direction_vector / norm
        else:
            norm = torch.norm(direction_vector)
            if norm > 0:
                direction_vector = direction_vector / norm

        eps_f = direction_vector * grounding_gap
        return eps_f

    def sample_context_drift(self, t: int, base_rate: Optional[float] = None) -> torch.Tensor:
        """
        Contextual drift caused by cumulative attention decay:
        Monotonically scales with execution step t.
        """
        rate = base_rate if base_rate is not None else self.drift_rate
        # Drift vector with soft random walk drift
        drift_mag = rate * (t ** 1.1)
        raw = torch.randn(self.dim)
        norm = torch.norm(raw)
        if norm > 0:
            eps_c = (raw / norm) * drift_mag
        else:
            eps_c = raw
        return eps_c

    def step(self, t: int, grounding_gap: float = 0.05, delta_i: Optional[float] = None) -> NoiseSample:
        """Generates unified tri-partite error at time step t."""
        eps_i = self.sample_intrinsic_noise(delta_i)
        eps_f = self.sample_feedback_distortion(grounding_gap)
        eps_c = self.sample_context_drift(t)
        
        total = eps_i + eps_f + eps_c
        return NoiseSample(
            total=total,
            intrinsic=eps_i,
            feedback=eps_f,
            drift=eps_c,
            norm_total=float(torch.norm(total).item()),
            norm_intrinsic=float(torch.norm(eps_i).item()),
            norm_feedback=float(torch.norm(eps_f).item()),
            norm_drift=float(torch.norm(eps_c).item())
        )
