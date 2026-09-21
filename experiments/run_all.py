"""
Master Orchestrator for All Experiments.
Runs Experiments 1 through 8 and outputs consolidated empirical data.
"""

import sys
import os
import argparse

# Ensure both Implementation directory and repo root are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

for path in [current_dir, parent_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from Implementation.experiments.exp1_error_accumulation import run_experiment_1
    from Implementation.experiments.exp2_spectral_analysis import run_experiment_2
    from Implementation.experiments.exp3_lyapunov_stability import run_experiment_3
    from Implementation.experiments.exp4_attention_saturation import run_experiment_4
    from Implementation.experiments.exp5_constraint_equation import run_experiment_5
    from Implementation.experiments.exp6_grounding_gap_drift import run_experiment_6
    from Implementation.experiments.exp7_ablation_benchmarks import run_experiment_7
    from Implementation.experiments.exp8_computational_profile import run_experiment_8
except ImportError:
    from exp1_error_accumulation import run_experiment_1
    from exp2_spectral_analysis import run_experiment_2
    from exp3_lyapunov_stability import run_experiment_3
    from exp4_attention_saturation import run_experiment_4
    from exp5_constraint_equation import run_experiment_5
    from exp6_grounding_gap_drift import run_experiment_6
    from exp7_ablation_benchmarks import run_experiment_7
    from exp8_computational_profile import run_experiment_8

def get_default_results_dir():
    # If parent is named Implementation, check if results exists there
    candidate = os.path.join(parent_dir, "results")
    if os.path.exists(candidate) or os.path.basename(parent_dir) == "Implementation":
        return candidate
    return "results"

def run_all(output_dir: str = None, test_only: bool = False):
    if output_dir is None:
        output_dir = get_default_results_dir()
        
    print("=" * 70)
    print("STARTING COMPLETE EMPIRICAL BENCHMARK SUITE")
    print("Closed-Loop Agent Dynamics & Stability Framework")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n>>> [1/8] Running Exp 1: Error Norm Accumulation & Confidence Ribbons...")
    run_experiment_1(output_dir)
    
    print("\n>>> [2/8] Running Exp 2: Spectral Radius Drift & Eigenvalue Spectra...")
    run_experiment_2(output_dir)
    
    print("\n>>> [3/8] Running Exp 3: Lyapunov Energy Dynamics & Phase Space Portraits...")
    run_experiment_3(output_dir)
    
    print("\n>>> [4/8] Running Exp 4: Attention Weight Saturation & Critical Horizon T*...")
    run_experiment_4(output_dir)
    
    print("\n>>> [5/8] Running Exp 5: Master Agentic Constraint Equation (Theorem 3)...")
    run_experiment_5(output_dir)
    
    print("\n>>> [6/8] Running Exp 6: Action Grounding Gap Sensitivity...")
    run_experiment_6(output_dir)
    
    print("\n>>> [7/8] Running Exp 7: 5-Architecture Long-Horizon Benchmark (Ablation)...")
    run_experiment_7(output_dir)
    
    print("\n>>> [8/8] Running Exp 8: Computational Complexity & Latency Profiling...")
    run_experiment_8(output_dir)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Artifacts and data stored in: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agent Stability Benchmarks")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save JSON benchmark results")
    parser.add_argument("--test-only", action="store_true", help="Run short test iterations")
    args = parser.parse_args()
    
    run_all(output_dir=args.output_dir, test_only=args.test_only)
