"""
Core mathematical constructs for Closed-Loop Agentic Dynamics.
"""
from .dynamics import ClosedLoopDynamicalSystem, TransformerContextUpdate
from .jacobian import compute_jacobian, compute_spectral_radius, compute_operator_norm, compute_jacobian_product_spectrum
from .lyapunov import LyapunovAnalyzer
from .noise import ErrorAccumulationPipeline, NoiseSample
from .attention import AttentionDecayProfile, compute_critical_horizon

__all__ = [
    "ClosedLoopDynamicalSystem",
    "TransformerContextUpdate",
    "compute_jacobian",
    "compute_spectral_radius",
    "compute_operator_norm",
    "compute_jacobian_product_spectrum",
    "LyapunovAnalyzer",
    "ErrorAccumulationPipeline",
    "NoiseSample",
    "AttentionDecayProfile",
    "compute_critical_horizon",
]
