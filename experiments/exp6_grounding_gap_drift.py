"""
Experiment 6: Action Grounding Gap Sensitivity and Environmental Misalignment.
Evaluates the impact of different grounding error regimes:
1. Nominal Gaussian
2. Temporal Drift
3. Heavy-tailed Pareto
4. Adversarial Oscillations
Quantifies the divergence dynamics under execution discrepancies Delta_g(t).
"""

import numpy as np
import torch
import os
import json
from ..config import NUM_MC_TRIALS, MAX_HORIZON_T, set_global_seed
from ..envs.orchestration_env import OrchestrationEnvironment
from ..agents.geometric_persistent_agent import GeometricPersistentAgent
from ..agents.prompt_wrapper_agent import PromptWrapperAgent
from ..agents.unregularized_agent import UnregularizedAgent

def run_experiment_6(output_dir: str = "Implementation/results"):
    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(42)
    
    T = MAX_HORIZON_T
    trials = 30
    gap_types = ["nominal", "drift", "pareto_heavy", "adversarial"]
    dim = 128
    
    regime_results = {}
    
    for g_type in gap_types:
        geo_dist = np.zeros((trials, T))
        prompt_dist = np.zeros((trials, T))
        unreg_dist = np.zeros((trials, T))
        
        geo_surv = 0
        prompt_surv = 0
        unreg_surv = 0
        
        for tr in range(trials):
            env_g = OrchestrationEnvironment(state_dim=dim, max_horizon=T, seed=tr*10)
            env_p = OrchestrationEnvironment(state_dim=dim, max_horizon=T, seed=tr*10)
            env_u = OrchestrationEnvironment(state_dim=dim, max_horizon=T, seed=tr*10)
            
            agent_g = GeometricPersistentAgent(dim=dim, seed=tr*10)
            agent_p = PromptWrapperAgent(dim=dim, seed=tr*10)
            agent_u = UnregularizedAgent(dim=dim, seed=tr*10)
            
            obs_g = env_g.reset()
            obs_p = env_p.reset()
            obs_u = env_u.reset()
            
            g_alive, p_alive, u_alive = True, True, True
            
            for t in range(T):
                # Geometric
                if g_alive:
                    a_g, info_g = agent_g.step(obs_g)
                    obs_g, _, done_g, env_info_g = env_g.step(a_g, gap_type=g_type)
                    geo_dist[tr, t] = env_info_g["dist_to_goal"]
                    if done_g and t < T - 1:
                        g_alive = False
                        geo_dist[tr, t+1:] = env_info_g["dist_to_goal"]
                
                # Prompt
                if p_alive:
                    a_p, info_p = agent_p.step(obs_p)
                    obs_p, _, done_p, env_info_p = env_p.step(a_p, gap_type=g_type)
                    prompt_dist[tr, t] = env_info_p["dist_to_goal"]
                    if done_p and t < T - 1:
                        p_alive = False
                        prompt_dist[tr, t+1:] = env_info_p["dist_to_goal"]
                        
                # Unregularized
                if u_alive:
                    a_u, info_u = agent_u.step(obs_u)
                    obs_u, _, done_u, env_info_u = env_u.step(a_u, gap_type=g_type)
                    unreg_dist[tr, t] = env_info_u["dist_to_goal"]
                    if done_u and t < T - 1:
                        u_alive = False
                        unreg_dist[tr, t+1:] = env_info_u["dist_to_goal"]
                        
            if g_alive: geo_surv += 1
            if p_alive: prompt_surv += 1
            if u_alive: unreg_surv += 1
            
        regime_results[g_type] = {
            "geometric_mean_dist": np.mean(geo_dist, axis=0).tolist(),
            "prompt_mean_dist": np.mean(prompt_dist, axis=0).tolist(),
            "unreg_mean_dist": np.mean(unreg_dist, axis=0).tolist(),
            "survival_rate": {
                "geometric": (geo_surv / trials) * 100.0,
                "prompt_wrapper": (prompt_surv / trials) * 100.0,
                "unregularized": (unreg_surv / trials) * 100.0
            }
        }
        
    out_file = os.path.join(output_dir, "exp6_grounding_gap_drift.json")
    with open(out_file, "w") as f:
        json.dump(regime_results, f, indent=2)
        
    print(f"[Exp 6 Complete] Grounding gap sensitivity saved to {out_file}")
    return regime_results

if __name__ == "__main__":
    run_experiment_6()
