"""
Experiment 4: Attention Saturation Dynamics, Weight Decay Profiles,
and Analytical Critical Horizon T*.
Evaluates historical attention attenuation and critical threshold T* across sequence lengths.
"""

import numpy as np
import os
import json
from ..core.attention import AttentionDecayProfile, compute_critical_horizon

def run_experiment_4(output_dir: str = "Implementation/results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Calibrated parameters yielding analytical critical horizon T* approx 60.02
    sigma_attn = 20.0
    delta = 0.01
    gamma = 1.1075  # Yields T* = 20 * sqrt(-2 * ln(0.011075)) = 20 * sqrt(9.006) = 60.02
    
    profile = AttentionDecayProfile(sigma_attn=sigma_attn, delta=delta, gamma=gamma)
    analytical_t_star = profile.t_star
    
    # 1. 1D slice profiles for t in {10, 30, 60, 90} across historical positions i in [1, 100]
    steps_to_evaluate = [10, 30, 60, 90]
    slices = {}
    for t_val in steps_to_evaluate:
        w = profile.compute_weights(t_val, max_tokens=100)
        slices[str(t_val)] = w.tolist()
        
    # 2. 2D Spatiotemporal Matrix [100, 100] for full heatmap visualization
    heatmap_2d = profile.get_saturation_profile_2d(max_t=100, max_i=100)
    
    # 3. Parametric sensitivity: sweep sigma_attn from 10 to 40 and prompt complexity gamma
    sigma_sweep = np.linspace(10, 40, 7).tolist()
    gamma_sweep = [0.8, 1.0, 1.1075, 1.3, 1.5]
    t_star_grid = {}
    for g in gamma_sweep:
        t_star_grid[str(g)] = [compute_critical_horizon(s, delta, g) for s in sigma_sweep]
        
    results = {
        "sigma_attn": sigma_attn,
        "delta": delta,
        "gamma": gamma,
        "analytical_t_star": analytical_t_star,
        "token_indices": list(range(1, 101)),
        "steps_evaluated": steps_to_evaluate,
        "profiles": slices,
        "heatmap_2d": heatmap_2d.tolist(),
        "parametric_sensitivity": {
            "sigma_sweep": sigma_sweep,
            "gamma_sweep": gamma_sweep,
            "t_star_grid": t_star_grid
        }
    }
    
    out_file = os.path.join(output_dir, "exp4_attention_saturation.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"[Exp 4 Complete] Analytical T* = {analytical_t_star:.2f}. Data saved to {out_file}")
    return results

if __name__ == "__main__":
    run_experiment_4()
