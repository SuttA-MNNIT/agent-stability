"""
Abstract Base Agent Interface for Closed-Loop Orchestration.
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional, Tuple

class BaseAgent(ABC):
    def __init__(self, name: str, dim: int = 128):
        self.name = name
        self.dim = dim
        self.h_t: Optional[torch.Tensor] = None
        self.step_count: int = 0

    @abstractmethod
    def reset(self, initial_state: Optional[torch.Tensor] = None):
        """Reset agent internal hidden state and step counter."""
        pass

    @abstractmethod
    def step(self, observation: torch.Tensor, grounding_gap: float = 0.05, compute_diagnostics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute one closed-loop step.
        Returns:
            action_vector: [d] output representation for tool execution
            info: dictionary containing diagnostic metrics (norm, spectral metrics, etc.)
        """
        pass
