"""
Agent architectures for Closed-Loop Orchestration.
"""
from .base_agent import BaseAgent
from .unregularized_agent import UnregularizedAgent
from .prompt_wrapper_agent import PromptWrapperAgent
from .geometric_persistent_agent import GeometricPersistentAgent
from .sliding_window_agent import SlidingWindowAgent
from .dnc_memory_agent import DNCMemoryAgent
from .ablation_agents import ProjectionOnlyAgent, DampingOnlyAgent

__all__ = [
    "BaseAgent",
    "UnregularizedAgent",
    "PromptWrapperAgent",
    "GeometricPersistentAgent",
    "SlidingWindowAgent",
    "DNCMemoryAgent",
    "ProjectionOnlyAgent",
    "DampingOnlyAgent"
]
