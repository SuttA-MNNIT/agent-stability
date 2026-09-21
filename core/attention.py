"""
Attention Saturation, Memory Decay, and Critical Horizon Analytics.
Implements:
alpha_i^{(t)} propto exp( - (t - i)^2 / (2 sigma_attn^2) ) * (1 / t)
T^* approx sigma_attn * sqrt( -2 ln( delta * gamma ) )
"""

import numpy as np
from typing import Tuple, Dict, Optional

def compute_critical_horizon(sigma_attn: float = 25.0, delta: float = 0.01, gamma: float = 1.2) -> float:
    """
    Evaluates Theorem 2:
    T^* = sigma_attn * sqrt( -2 * ln(delta * gamma) )
    """
    arg = delta * gamma
    if arg <= 0 or arg >= 1.0:
        raise ValueError(f"Invalid argument to logarithm: delta * gamma = {arg}, must be in (0, 1).")
    t_star = sigma_attn * np.sqrt(-2.0 * np.log(arg))
    return float(t_star)

class AttentionDecayProfile:
    """
    Models temporal attention weight distribution over historical context indices i in [1, t].
    """
    def __init__(self, sigma_attn: float = 25.0, delta: float = 0.01, gamma: float = 1.2):
        self.sigma_attn = sigma_attn
        self.delta = delta
        self.gamma = gamma
        self.t_star = compute_critical_horizon(sigma_attn, delta, gamma)

    def compute_weights(self, t: int, max_tokens: Optional[int] = None) -> np.ndarray:
        """
        Computes attention weight vector alpha^{(t)} across historical positions i in [1, t].
        If max_tokens is provided, zero-pads up to max_tokens for consistent plotting.
        """
        if t < 1:
            return np.array([1.0])
            
        indices = np.arange(1, t + 1)
        # Spatial-temporal exponential kernel
        exponent = -((t - indices) ** 2) / (2.0 * (self.sigma_attn ** 2))
        raw_weights = np.exp(exponent) / float(t)
        
        # Softmax / simplex normalization
        sum_weights = np.sum(raw_weights)
        if sum_weights > 0:
            weights = raw_weights / sum_weights
        else:
            weights = np.ones_like(raw_weights) / len(raw_weights)
            
        if max_tokens is not None and max_tokens > t:
            padded = np.zeros(max_tokens)
            padded[:t] = weights
            return padded
            
        return weights

    def compute_historical_retention(self, t: int, initial_token_idx: int = 1) -> float:
        """
        Returns the retention weight of the foundational initial prompt token alpha_1^{(t)}.
        """
        weights = self.compute_weights(t)
        if initial_token_idx <= len(weights):
            return float(weights[initial_token_idx - 1])
        return 0.0

    def get_saturation_profile_2d(self, max_t: int = 100, max_i: int = 100) -> np.ndarray:
        """
        Generates 2D matrix [max_t, max_i] of attention weights across all (t, i) pairs.
        Used for 2D context saturation heatmaps.
        """
        grid = np.zeros((max_t, max_i))
        for t in range(1, max_t + 1):
            w = self.compute_weights(t, max_tokens=max_i)
            grid[t - 1, :] = w
        return grid
