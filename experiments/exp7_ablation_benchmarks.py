"""
Experiment 7: Comprehensive Multi-Architecture Benchmark and Ablation.
Evaluates 5 agent architectures across horizons T in {10, 25, 50, 75, 100, 150}:
1. Unregularized Vanilla Loop
2. Prompt-based Wrapper (ReAct/Reflexion)
3. Sliding Window (W=16)
4. External Memory Bank (DNC)
5. Geometric Memory Persistence (Ours)
Evaluates trajectory-boundedness, spectral drift, and failure rates over long execution horizons.
"""

import numpy as np
import torch
import os
import json
from ..config import set_global_seed
from ..envs.orchestration_env import OrchestrationEnvironment
from ..agents.unregularized_agent import UnregularizedAgent
from ..agents.prompt_wrapper_agent import PromptWrapperAgent
from ..agents.sliding_window_agent import SlidingWindowAgent
from ..agents.dnc_memory_agent import DNCMemoryAgent
from ..agents.geometric_persistent_agent import GeometricPersistentAgent
from ..agents.ablation_agents import ProjectionOnlyAgent, DampingOnlyAgent

def run_experiment_7(output_dir: str = "Implementation/results"):
    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(42)
    
    horizons = [10, 25, 50, 75, 100, 150]
    num_trials = 50
    dim = 128
    
    agent_classes = {
        "Unregularized": lambda s: UnregularizedAgent(dim=dim, seed=s),
        "Prompt Wrapper": lambda s: PromptWrapperAgent(dim=dim, seed=s),
        "Sliding Window (W=16)": lambda s: SlidingWindowAgent(dim=dim, window_size=16, seed=s),
        "External Memory (DNC)": lambda s: DNCMemoryAgent(dim=dim, mem_slots=32, seed=s),
        "Projection-Only (No Damping)": lambda s: ProjectionOnlyAgent(dim=dim, seed=s),
        "Damping-Only (No Projection)": lambda s: DampingOnlyAgent(dim=dim, seed=s),
        "Geometric Persistence (Ours)": lambda s: GeometricPersistentAgent(dim=dim, seed=s)
    }
    
    # Structure for Table II: Success rate across horizons
    table2_success_rates = {name: {} for name in agent_classes}
    
    # Structure for Table III: Detailed stability metrics at T=100
    table3_metrics = {name: {} for name in agent_classes}
    
    for T in horizons:
        print(f"  Evaluating horizon T = {T}...")
        for name, agent_fn in agent_classes.items():
            successes = 0
            failures = 0
            divergences = 0
            ttfs = []
            max_rhos = []
            
            for tr in range(num_trials):
                seed = 42 + tr * 13
                env = OrchestrationEnvironment(state_dim=dim, max_horizon=T, seed=seed)
                agent = agent_fn(seed)
                obs = env.reset(seed=seed)
                
                # Nominal error-free reference state h_bar
                h_bar = agent.h_t.clone()
                nominal_model = agent.model
                
                alive = True
                failure_step = T
                trial_rhos = []
                diverged = False
                
                for t in range(T):
                    act, info = agent.step(obs)
                    obs, _, done, env_info = env.step(act, gap_type="nominal")
                    trial_rhos.append(info.get("spectral_radius", 0.8))
                    
                    # Update nominal state using agent's unperturbed dynamics
                    with torch.no_grad():
                        h_bar_nom = nominal_model(h_bar, obs)
                        if "Geometric" in name:
                            n = torch.norm(h_bar_nom)
                            h_bar = (h_bar_nom / n) * 1.5 if n > 1.5 else h_bar_nom
                        else:
                            h_bar = h_bar_nom
                        
                    # Deviation from nominal trajectory Delta h_t = h_t - h_bar
                    delta_h = float(torch.norm(agent.h_t - h_bar).item())
                    
                    # Critical divergence check
                    if delta_h > 2.2 or env_info["is_diverged"]:
                        divergences += 1
                        diverged = True
                        alive = False
                        failure_step = t + 1
                        break
                    elif delta_h > 0.85:
                        # Functional action error / grounding failure threshold
                        alive = False
                        failure_step = t + 1
                        break
                        
                if alive and not diverged:
                    successes += 1
                else:
                    failures += 1
                    
                ttfs.append(failure_step)
                max_rhos.append(max(trial_rhos) if trial_rhos else 1.0)
                
            sr = (successes / num_trials) * 100.0
            p = successes / num_trials
            z = 1.96
            n = num_trials
            denom = 1.0 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            margin = (z * np.sqrt((p * (1.0 - p) / n) + (z**2 / (4 * n**2)))) / denom
            ci_low = max(0.0, float((center - margin) * 100.0))
            ci_high = min(100.0, float((center + margin) * 100.0))
            
            table2_success_rates[name][str(T)] = {
                "rate": round(sr, 1),
                "ci_95": [round(ci_low, 1), round(ci_high, 1)]
            }
            
            # Record detailed metrics at benchmark horizon T=100
            if T == 100:
                ttf_mean = float(np.mean(ttfs))
                ttf_sem = float(np.std(ttfs, ddof=1) / np.sqrt(n))
                peak_mean = float(np.mean(max_rhos))
                peak_sem = float(np.std(max_rhos, ddof=1) / np.sqrt(n))
                
                table3_metrics[name] = {
                    "success_rate": round(sr, 1),
                    "success_ci": [round(ci_low, 1), round(ci_high, 1)],
                    "mean_ttf": round(ttf_mean, 1),
                    "mean_ttf_sem": round(ttf_sem, 2),
                    "divergence_rate": round((divergences / num_trials) * 100.0, 1),
                    "peak_rho": round(peak_mean, 3),
                    "peak_rho_sem": round(peak_sem, 3),
                    "stability_margin": round(float(1.0 - peak_mean), 3) if peak_mean < 1.0 else 0.000
                }
                
    results = {
        "num_trials": num_trials,
        "horizons": horizons,
        "table2_success_rates": table2_success_rates,
        "table3_stability_metrics": table3_metrics
    }
    
    out_file = os.path.join(output_dir, "exp7_ablation_benchmarks.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"[Exp 7 Complete] Multi-architecture benchmarks saved to {out_file}")
    return results

if __name__ == "__main__":
    run_experiment_7()
