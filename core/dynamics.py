"""
Discrete-time Stochastic Dynamical System & Transformer Context Update Operator.
Implements: h_{t+1} = f_theta(h_t, o_t) + epsilon_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class TransformerContextUpdate(nn.Module):
    """
    Parametric Transformer Context Update Operator f_theta(h_t, o_t).
    Models multi-head attention projection, residual connections,
    and MLP feedforward layer with LayerNorm.
    """
    def __init__(self, dim: int = 128, num_heads: int = 4, mlp_ratio: float = 2.0, spectral_bound: Optional[float] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.spectral_bound = spectral_bound
        
        # Linear projections for Query, Key, Value from hidden state and observation
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        
        # Observation injection projection
        self.w_obs = nn.Linear(dim, dim, bias=False)
        
        # MLP Block
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Orthogonal / Xavier initialization for controlled spectral properties."""
        nn.init.orthogonal_(self.w_q.weight, gain=0.8)
        nn.init.orthogonal_(self.w_k.weight, gain=0.8)
        nn.init.orthogonal_(self.w_v.weight, gain=0.8)
        nn.init.orthogonal_(self.w_o.weight, gain=0.8)
        nn.init.orthogonal_(self.w_obs.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.7)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.7)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h_t: torch.Tensor, o_t: torch.Tensor) -> torch.Tensor:
        """
        Computes nominal state transition:
        h_{t+1} = f_theta(h_t, o_t)
        Args:
            h_t: [d] or [batch, d] hidden state vector
            o_t: [d] or [batch, d] environment observation vector
        Returns:
            h_{t+1}: [d] or [batch, d] next hidden state vector
        """
        # Ensure batch dimension
        is_1d = (h_t.dim() == 1)
        if is_1d:
            h_t = h_t.unsqueeze(0)
            o_t = o_t.unsqueeze(0)
            
        # Observation fusion
        obs_feat = self.w_obs(o_t)
        h_norm1 = self.ln1(h_t)
        
        # Multi-head self-attention simulation on [h_norm1, obs_feat]
        # Query from h_t, Keys/Values from combined [h_t, obs_feat]
        q = self.w_q(h_norm1).view(-1, self.num_heads, self.head_dim)
        k_h = self.w_k(h_norm1).view(-1, self.num_heads, self.head_dim)
        k_o = self.w_k(obs_feat).view(-1, self.num_heads, self.head_dim)
        v_h = self.w_v(h_norm1).view(-1, self.num_heads, self.head_dim)
        v_o = self.w_v(obs_feat).view(-1, self.num_heads, self.head_dim)
        
        # Keys and Values stacked: [batch, heads, 2, head_dim]
        k = torch.stack([k_h, k_o], dim=2)
        v = torch.stack([v_h, v_o], dim=2)
        
        # Attention scores: q * k^T / sqrt(d_k) -> [batch, heads, 1, 2]
        scores = torch.einsum('bhd,bhkd->bhk', q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # Context vector: [batch, heads, head_dim] -> [batch, dim]
        context = torch.einsum('bhk,bhkd->bhd', attn, v).reshape(-1, self.dim)
        attn_out = self.w_o(context)
        
        # Contractive Residual 1
        h_mid = 0.75 * h_t + 0.25 * attn_out
        
        # MLP block with GELU & Contractive Residual 2
        h_norm2 = self.ln2(h_mid)
        mlp_out = self.fc2(F.gelu(self.fc1(h_norm2)))
        h_next = 0.75 * h_mid + 0.25 * mlp_out
        
        # Optional spectral contraction enforcement
        if self.spectral_bound is not None:
            norm = torch.norm(h_next, dim=-1, keepdim=True)
            scale = torch.clamp(norm / self.spectral_bound, min=1.0)
            h_next = h_next / scale
            
        if is_1d:
            return h_next.squeeze(0)
        return h_next


class ClosedLoopDynamicalSystem:
    """
    Manages state-space trajectory generation over execution horizon T:
    h_{t+1} = f_theta(h_t, o_t) + epsilon_t
    """
    def __init__(self, model: TransformerContextUpdate, state_dim: int = 128):
        self.model = model
        self.state_dim = state_dim
        
    def step(self, h_t: torch.Tensor, o_t: torch.Tensor, epsilon_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single closed-loop update."""
        with torch.no_grad():
            h_nom = self.model(h_t, o_t)
            if epsilon_t is not None:
                return h_nom + epsilon_t
            return h_nom

    def simulate_trajectory(self, h_0: torch.Tensor, observations: torch.Tensor, noise_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simulate state sequence over T steps.
        Args:
            h_0: [d] initial language vector
            observations: [T, d] observation sequence
            noise_seq: [T, d] optional error sequence
        Returns:
            trajectory: [T+1, d] state vectors across time
        """
        T = observations.shape[0]
        traj = [h_0]
        h_curr = h_0
        for t in range(T):
            eps = noise_seq[t] if noise_seq is not None else None
            h_curr = self.step(h_curr, observations[t], eps)
            traj.append(h_curr)
        return torch.stack(traj, dim=0)
