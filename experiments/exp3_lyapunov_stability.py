"""
Experiment 3: Lyapunov Energy Dynamics & Phase Space State Trajectories.
Simulates quadratic energy evolution V(h) = h^T P h under contractive vs diverging control loops.
"""

import numpy as np
import torch
import os
import json
from ..config import MAX_HORIZON_T, set_global_seed
from ..core.lyapunov import LyapunovAnalyzer
from ..agents.geometric_persistent_agent import GeometricPersistentAgent
from ..agents.unregularized_agent import UnregularizedAgent

def run_experiment_3(output_dir: str = "Implementation/results"):
    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(42)
    
    T = MAX_HORIZON_T
    dim = 128
    
    # 1. Condition number sweep on Lyapunov Analyzer
    kappas = [1.0, 5.0, 20.0, 50.0]
    lyap_curves = {}
    
    for kappa in kappas:
        analyzer = LyapunovAnalyzer(dim=dim, condition_number=kappa, seed=42)
        unreg = UnregularizedAgent(dim=dim, seed=42)
        geo = GeometricPersistentAgent(dim=dim, seed=42)
        
        unreg_v = []
        geo_v = []
        
        for t in range(T):
            obs = torch.randn(dim) * 0.1
            _, info_u = unreg.step(obs, grounding_gap=0.05)
            _, info_g = geo.step(obs, grounding_gap=0.05)
            
            v_u = analyzer.compute_energy(info_u["state"])
            v_g = analyzer.compute_energy(info_g["state"])
            
            unreg_v.append(v_u)
            geo_v.append(v_g)
            
        lyap_curves[f"kappa_{int(kappa)}"] = {
            "unregularized": unreg_v,
            "geometric": geo_v
        }
        
    # 2. Extract 2D/3D phase space coordinates for Phase Portrait Inset
    unreg_agent = UnregularizedAgent(dim=dim, seed=101)
    geo_agent = GeometricPersistentAgent(dim=dim, seed=101)
    
    unreg_traj = []
    geo_traj = []
    
    for t in range(T):
        obs = torch.randn(dim) * 0.1
        _, info_u = unreg_agent.step(obs, grounding_gap=0.05)
        _, info_g = geo_agent.step(obs, grounding_gap=0.05)
        
        # Take first 3 coordinates for phase portrait
        unreg_traj.append(info_u["state"][:3].cpu().numpy().tolist())
        geo_traj.append(info_g["state"][:3].cpu().numpy().tolist())

    results = {
        "horizon": list(range(1, T + 1)),
        "lyapunov_energy_by_kappa": lyap_curves,
        "phase_space": {
            "unregularized": unreg_traj,
            "geometric": geo_traj
        }
    }
    
    out_file = os.path.join(output_dir, "exp3_lyapunov_stability.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"[Exp 3 Complete] Lyapunov & Phase space data saved to {out_file}")
    return results

if __name__ == "__main__":
    run_experiment_3()
