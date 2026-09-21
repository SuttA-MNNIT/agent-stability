"""
Global Configuration and Hyperparameters for
Closed-Loop Agent Dynamics & Stability Benchmarks
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import torch

# Random Seeds for Reproducibility
DEFAULT_SEED = 42
NUM_MC_TRIALS = 50
MAX_HORIZON_T = 100

# Architectural Dimensions
DEFAULT_STATE_DIM = 128
DIMENSION_SWEEP = [32, 64, 128, 256, 512]

# Attention Parameters
DEFAULT_SIGMA_ATTN = 25.0
DEFAULT_FIDELITY_THRESHOLD_DELTA = 0.01
DEFAULT_PROMPT_COMPLEXITY_GAMMA = 1.2

# Noise Settings
NOISE_LEVELS = {
    "low": 0.02,
    "medium": 0.08,
    "high": 0.18,
    "critical": 0.25
}

# Grounding Gap Settings
GROUNDING_GAP_NOMINAL = 0.05
GROUNDING_GAP_DRIFT_RATE = 0.008

# Lyapunov Parameters
LYAPUNOV_DISSIPATION_GAMMA = 0.98
DEFAULT_STABILITY_MARGIN_DELTA = 0.15

# Visual Palette (IEEE / Nature Inspired Colors)
COLOR_PALETTE = {
    "indigo": "#2563EB",       # Geometric Persistence / Stable
    "teal": "#0D9488",         # Attractor / Sink
    "crimson": "#DC2626",      # Unregularized / Drift / Divergence
    "amber": "#D97706",        # Prompt Wrapper
    "purple": "#7C3AED",       # DNC Memory Bank
    "slate": "#475569",        # Sliding Window / Neutral
    "emerald": "#059669",      # Bound
    "charcoal": "#1E293B"      # Text / Axes
}

def set_global_seed(seed: int = DEFAULT_SEED):
    """Ensure strict deterministic reproducibility across NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
