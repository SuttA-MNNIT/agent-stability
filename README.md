# Agent-Stability: Bounded Dynamics & Geometric Memory Persistence for Autonomous LLM Agents

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stability](https://img.shields.io/badge/Dynamics-Contractive-success.svg)](#theoretical-framework)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#quickstart)

A formal dynamical systems and empirical benchmarking framework for analyzing **error accumulation**, **context saturation**, and **geometric contractivity** in closed-loop Large Language Model (LLM) agent orchestration.

---

## Overview

When LLM-based autonomous agents operate in continuous, closed-loop execution environments—iteratively generating actions, observing noisy environmental feedback, and updating their internal context vector—small perturbations compound across steps. This leads to **silent degradation**, characterized by repetitive task loops, hallucinated tool invocations, and catastrophic plan derailment.

This repository provides:
1. **Mathematical State-Space Recurrence Framework**: Formulates the agentic hidden state evolution as a discrete-time stochastic dynamical system with exact Autograd Jacobian tracking and Lyapunov energy analysis.
2. **Tri-Partite Noise Modeling**: Decomposes closed-loop error into three orthogonal vectors: intrinsic sampling noise $\boldsymbol{\epsilon}_t^{(i)}$, effector grounding discrepancies $\boldsymbol{\epsilon}_t^{(f)}$, and historical attention dispersion $\boldsymbol{\epsilon}_t^{(c)}$.
3. **Geometric Memory Persistence**: Implements an active stabilization mechanism that projects internal representations onto the orthogonal Stiefel manifold ($\mathrm{Proj}_{\mathrm{Stiefel}}(\mathbf{h}_t) = \mathbf{U}\mathbf{V}^\top$) combined with Lyapunov dissipation damping ($\gamma \in (0, 1)$), provably guaranteeing contractive state trajectories ($\rho(\mathbf{J}_t) \le 1 - \eta$).
4. **Standardized 5-Architecture Benchmark Suite**: Comprehensive Monte Carlo evaluation ($N=50$ trials) across horizons up to $T = 150$ steps, comparing:
   - **Vanilla Unregularized** (unconstrained autoregressive loop)
   - **Prompt-Based Wrappers** (ReAct / Reflexion)
   - **Sliding Window Truncation** ($W = 16$)
   - **External Matrix Memory** (Differentiable Neural Computer / DNC)
   - **Geometric Memory Persistence** (Ours)

---

## Architectural Workflow

```
       +-----------------------------------------------------------------------+
       |             Tri-Partite Error Decomposition Pipeline                  |
       |  [ Intrinsic Sampling ]      [ Context Drift ]      [ Feedback Noise ]|
       |       epsilon_t^(i)        +   epsilon_t^(c)      +   epsilon_t^(f)   |
       +-----------------------------------------------------------+-----------+
                                                                   |
                                                      Additive Noise epsilon_t
                                                                   v
     +-------------------+   Contractive h_(t+1)   +---------------------------+
     | Agent State       |<------------------------| Context Update Block      |
     | Vector  h_t in R^d|                         | h_tilde = f_theta + eps_t |
     +---------+---------+                         +-------------+-------------+
               |                                                 ^
          Latent State h_t                                Observation o_t
               v                                                 |
     +---------+---------+   Action a_t^text       +-------------+-------------+
     | Action Policy     |------------------------>| External Environment      |
     | pi_theta(h_t)     |   (via Grounding Gap)   | s_(t+1) ~ T(s_t, a_t^env) |
     +-------------------+                         +---------------------------+
```

---

## Key Benchmark Findings

All experiments evaluated over $N = 50$ Monte Carlo trials across horizons up to $T = 150$ steps:

| Architecture Paradigm | State Recurrence | Stability Guarantee | Spectral Radius $\rho(\mathbf{J}_t)$ | Trajectory Boundedness ($T=100$) | Trajectory Boundedness ($T=150$) | Mean Time to Failure (MTTF) | Memory Footprint | Per-Step Overhead |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Autoregressive** | Unconstrained | None | Unbounded | 10.0% | 0.0% | 51.4 steps | $\mathcal{O}(T)$ | Baseline (0.91 ms) |
| **Prompt Wrapper (ReAct/Reflexion)** | Prompt text appending | None (Heuristic) | $\rho > 1.0$ at $t \approx 47$ | 16.0% | 0.0% | 59.2 steps | $\mathcal{O}(T)$ (14.8 MB) | +12% |
| **Sliding Window ($W=16$)** | Truncated context | Local bounded norm | $\rho \approx 0.850$ | 30.0% | 0.0% | 78.8 steps | $\mathcal{O}(W)$ (11.2 MB) | +6% |
| **External Memory (DNC)** | Content-addressed matrix | Heuristic retrieval | $\rho > 1.0$ (Peak 1.068) | 26.0% | 6.0% | 65.6 steps | $\mathcal{O}(N \times d)$ (18.2 MB) | +48% |
| **Geometric Persistence (Ours)** | **Stiefel Manifold + Damping** | **Provably Contractive** | **$\rho \le 0.82 \pm 0.01$** | **86.0%** | **70.0%** | **97.9 steps** | **$\mathcal{O}(1)$ (8.6 MB)** | **+18% (1.06 ms)** |

---

## Directory Structure

```
.
├── config.py                          # Global hyperparameters, dimensions, and random seeds
├── requirements.txt                   # Dependency specifications
├── pyproject.toml                     # Modern package installation metadata
├── LICENSE                            # MIT Open Source License
├── README.md                          # Documentation and quickstart
├── core/                              # Foundational mathematical modules
│   ├── dynamics.py                    # Multi-head Transformer context update f_theta(h_t, o_t)
│   ├── jacobian.py                    # Exact Autograd Jacobian, spectral radius, & operator norm
│   ├── lyapunov.py                    # Symmetric positive-definite energy analyzer V(h) = h^T P h
│   ├── noise.py                       # Tri-partite error pipeline: epsilon^(i) + epsilon^(f) + epsilon^(c)
│   └── attention.py                   # Gaussian/exponential attention decay profile & critical T*
├── agents/                            # Autonomous agent implementations
│   ├── base_agent.py                  # Abstract Agent Base Class
│   ├── unregularized_agent.py         # Vanilla open-loop autoregressive agent
│   ├── prompt_wrapper_agent.py        # Prompt-based wrapper (ReAct / Reflexion proxy)
│   ├── geometric_persistent_agent.py  # Proposed Geometric Memory Persistence agent
│   ├── sliding_window_agent.py        # Context window truncation baseline (W=16)
│   ├── dnc_memory_agent.py            # Differentiable Neural Computer external memory baseline
│   └── ablation_agents.py             # Projection-Only and Damping-Only ablation variants
├── envs/                              # Simulation environment
│   └── orchestration_env.py           # Multi-step tool-use environment with grounding gap Delta_g(t)
├── experiments/                       # Reproducible Monte Carlo evaluation scripts
│   ├── exp1_error_accumulation.py     # Error norm accumulation & confidence ribbons
│   ├── exp2_spectral_analysis.py      # Spectral radius drift & complex eigenvalue spectrum
│   ├── exp3_lyapunov_stability.py     # Lyapunov energy trajectories & phase space portraits
│   ├── exp4_attention_saturation.py   # Context saturation profiles & analytical T* validation
│   ├── exp5_constraint_equation.py    # Master Agentic Constraint Equation (dimensional fragility)
│   ├── exp6_grounding_gap_drift.py    # Action grounding gap sensitivity under 4 noise regimes
│   ├── exp7_ablation_benchmarks.py    # Multi-architecture ablation benchmarks
│   ├── exp8_computational_profile.py  # Latency, memory footprint, & FLOPs profiling
│   └── run_all.py                     # Master execution runner
└── visualization/                     # High-resolution plotting and diagnostics
    ├── plot_styles.py                 # Clean publication plot style configuration
    └── generate_all_figures.py        # Compiles all simulation artifacts into PDFs and PNGs
```

---

## Quickstart

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/<your-username>/agent-stability.git
cd agent-stability
pip install -r requirements.txt
```

Alternatively, install in editable mode:
```bash
pip install -e .
```

### 2. Run the Benchmark Suite

To execute all 8 Monte Carlo experiments:

```bash
python experiments/run_all.py
```

Results will be automatically stored as structured JSON files in `results/`.

### 3. Generate Publication Figures

To compile vector PDF and raster PNG figures from experimental data:

```bash
python visualization/generate_all_figures.py
```

Generated plots will be saved to `figures/`.

---

## Theoretical Framework

### 1. Discrete-Time State Recurrence
The high-dimensional latent language vector evolves according to:
$$\mathbf{h}_{t+1} = f_\theta(\mathbf{h}_t, o_t) + \boldsymbol{\epsilon}_t$$
where $\boldsymbol{\epsilon}_t = \boldsymbol{\epsilon}_t^{(i)} + \boldsymbol{\epsilon}_t^{(f)} + \boldsymbol{\epsilon}_t^{(c)}$.

### 2. Agentic Stability Bound
For a discrete-time Lyapunov candidate $V(\mathbf{h}_t) = \mathbf{h}_t^\top \mathbf{P} \mathbf{h}_t$, asymptotic trajectory boundedness requires:
$$\rho\left(\prod_{k=0}^{T-1} \mathbf{J}_k\right) < 1 - \frac{\bar{\epsilon}}{\lambda_{\min}(\mathbf{P})}$$
where $\mathbf{J}_k = \nabla_{\mathbf{h}} f_\theta(\mathbf{h}_k, o_k)$ is the transition Jacobian.

### 3. Geometric Memory Persistence
At each context update, internal representations are projected onto the orthogonal Stiefel manifold $\mathrm{St}(d, d) = \{\mathbf{Q} \in \mathbb{R}^{d \times d} : \mathbf{Q}^\top \mathbf{Q} = \mathbf{I}\}$ via polar decomposition:
$$\mathbf{h}_{t+1} = \gamma \cdot \mathrm{Proj}_{\mathrm{Stiefel}}(\widetilde{\mathbf{h}}_{t+1}) = \gamma \cdot \mathbf{U}\mathbf{V}^\top \quad \text{where} \quad \widetilde{\mathbf{h}}_{t+1} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$
Dissipation scalar $\gamma \in (0, 1)$ guarantees strict contractivity: $\|\mathbf{J}_t\|_{\mathrm{op}} \le 1 - \eta$.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
