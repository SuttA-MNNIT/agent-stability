"""
Experiment 2: Spectral Radius Dynamics, Critical Breach Detection,
and Complex Plane Eigenvalue Spectra Tracking.
"""

import numpy as np
import torch
import os
import json
from ..config import MAX_HORIZON_T, set_global_seed
from ..agents.geometric_persistent_agent import GeometricPersistentAgent
from ..agents.prompt_wrapper_agent import PromptWrapperAgent
from ..agents.unregularized_agent import UnregularizedAgent
from ..core.jacobian import compute_eigenvalue_spectrum

def run_experiment_2(output_dir: str = "Implementation/results"):
    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(42)
    
    T = MAX_HORIZON_T
    dim = 128
    
    geo_agent = GeometricPersistentAgent(dim=dim, seed=42)
    prompt_agent = PromptWrapperAgent(dim=dim, seed=42)
    unreg_agent = UnregularizedAgent(dim=dim, seed=42)
    
    geo_rho = []
    prompt_rho = []
    unreg_rho = []
    
    # Store eigenvalue spectra at specific snapshot checkpoints
    snapshots = [10, 30, 50, 70, 90]
    eigen_snapshots = {
        "snapshots": snapshots,
        "prompt_wrapper": {},
        "geometric": {}
    }
    
    for t in range(T):
        obs = torch.randn(dim) * 0.1
        step_num = t + 1
        is_snap = (step_num in snapshots)
        
        _, info_geo = geo_agent.step(obs, grounding_gap=0.05, compute_diagnostics=is_snap)
        _, info_prompt = prompt_agent.step(obs, grounding_gap=0.05, compute_diagnostics=is_snap)
        _, info_unreg = unreg_agent.step(obs, grounding_gap=0.05, compute_diagnostics=is_snap)
        
        geo_rho.append(float(info_geo["spectral_radius"]))
        prompt_rho.append(float(info_prompt["spectral_radius"]))
        unreg_rho.append(float(info_unreg["spectral_radius"]))
        
        if is_snap:
            real_p, imag_p = compute_eigenvalue_spectrum(info_prompt["jacobian"])
            real_g, imag_g = compute_eigenvalue_spectrum(info_geo["jacobian"])
            
            eigen_snapshots["prompt_wrapper"][str(step_num)] = {
                "real": real_p.tolist(),
                "imag": imag_p.tolist()
            }
            eigen_snapshots["geometric"][str(step_num)] = {
                "real": real_g.tolist(),
                "imag": imag_g.tolist()
            }

    # Find exact step where prompt wrapper breaches 1.0
    crossing_step = None
    for idx, val in enumerate(prompt_rho):
        if val >= 1.0:
            crossing_step = idx + 1
            break

    results = {
        "horizon": list(range(1, T + 1)),
        "geometric_rho": geo_rho,
        "prompt_wrapper_rho": prompt_rho,
        "unregularized_rho": unreg_rho,
        "crossing_step": crossing_step,
        "eigenvalues": eigen_snapshots
    }
    
    out_file = os.path.join(output_dir, "exp2_spectral_analysis.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"[Exp 2 Complete] Crossing at step {crossing_step}. Data saved to {out_file}")
    return results

if __name__ == "__main__":
    run_experiment_2()
