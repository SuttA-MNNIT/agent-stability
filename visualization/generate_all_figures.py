"""
Comprehensive Figure Generation Pipeline.
Renders all 9 publication-grade IEEE figures in both vector PDF and raster PNG:
- fig2_error_norm
- fig3_spectral_radius
- fig4_lyapunov
- fig5_attention
- fig6_eigenvalue_spectra
- fig7_constraint_surface
- fig8_grounding_sensitivity
- fig9_ablation_multi_metric
- fig10_attention_heatmap
"""

import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from Implementation.visualization.plot_styles import apply_ieee_style, COLORS
except ImportError:
    from plot_styles import apply_ieee_style, COLORS

apply_ieee_style()

def plot_fig2_error_norm(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    T = len(data["low"]["mean"])
    x = np.arange(1, T + 1)
    
    # Palette mapping
    color_map = {
        "low": COLORS["teal"],
        "medium": COLORS["amber"],
        "high": COLORS["crimson"],
        "critical": "#7F1D1D"
    }
    label_map = {
        "low": r"Low Noise ($\delta_i = 0.02$)",
        "medium": r"Medium Noise ($\delta_i = 0.08$)",
        "high": r"High Noise ($\delta_i = 0.18$)",
        "critical": r"Critical Noise ($\delta_i = 0.25$)"
    }
    
    for regime in ["low", "medium", "high", "critical"]:
        mean = np.array(data[regime]["mean"])
        lower = np.array(data[regime]["ci_lower"])
        upper = np.array(data[regime]["ci_upper"])
        c = color_map[regime]
        
        ax.plot(x, mean, label=label_map[regime], color=c, lw=2.0)
        ax.fill_between(x, lower, upper, color=c, alpha=0.18)
        
    # Mark Critical Horizon T* = 60
    ax.axvline(x=60, color=COLORS["crimson"], linestyle="--", lw=1.5, alpha=0.85)
    ax.text(61, ax.get_ylim()[1] * 0.75, r"Critical Horizon $T^* = 60$", color=COLORS["crimson"],
            fontweight="bold", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec=COLORS["crimson"], alpha=0.85))
    
    # Mark theoretical stability margin
    ax.axhline(y=18.0, color=COLORS["slate"], linestyle=":", lw=1.3, label="Theoretical Stability Margin")
    
    ax.set_xlabel("Execution Horizon $t$ (Steps)")
    ax.set_ylabel(r"Cumulative Error Norm $\|\boldsymbol{\epsilon}_t\|$")
    ax.set_title(r"Cumulative Error Norm Dynamics across Noise Regimes ($95\%$ CI, $N=50$)")
    ax.set_xlim(1, T)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(loc="upper left")
    
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "fig2_error_norm.pdf"))
        fig.savefig(os.path.join(d, "fig2_error_norm.png"))
    plt.close(fig)
    print("  [Saved] fig2_error_norm")

def plot_fig3_spectral_radius(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.array(data["horizon"])
    geo_rho = np.array(data["geometric_rho"])
    prompt_rho = np.array(data["prompt_wrapper_rho"])
    unreg_rho = np.array(data["unregularized_rho"])
    crossing = data["crossing_step"]
    
    ax.plot(x, geo_rho, label="Geometric Memory Persistence (Ours)", color=COLORS["indigo"], lw=2.2)
    ax.plot(x, prompt_rho, label="Prompt-based Behavioral Wrapper", color=COLORS["crimson"], lw=2.0)
    ax.plot(x, unreg_rho, label="Unregularized Autoregressive Loop", color=COLORS["amber"], lw=1.8, linestyle="-.")
    
    # Stability threshold line rho = 1.0
    ax.axhline(y=1.0, color="#1E293B", linestyle="--", lw=1.5, label=r"Stability Threshold $\rho(\mathbf{J}_t) = 1.0$")
    
    if crossing:
        ax.plot([crossing], [1.0], marker='o', markersize=8, color=COLORS["crimson"], zorder=5)
        ax.annotate(r"Unit Circle Crossing" + "\n" + rf"($t \approx {crossing}$)",
                    xy=(crossing, 1.0), xytext=(19, 1.31),
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=COLORS["crimson"], lw=1.3,
                                    connectionstyle="arc3,rad=-0.12"),
                    fontweight="bold", fontsize=8.2,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec=COLORS["crimson"], alpha=0.95))
        
    ax.set_xlabel("Execution Horizon $t$ (Steps)")
    ax.set_ylabel(r"Spectral Radius $\rho(\mathbf{J}_t)$")
    ax.set_title(r"Spectral Radius Dynamics and Stability Threshold Violation")
    ax.set_xlim(1, len(x))
    ax.set_ylim(0.4, 1.50)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8.5)
    
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig3_spectral_radius.pdf"))
        fig.savefig(os.path.join(d, "fig3_spectral_radius.png"))
    plt.close(fig)
    print("  [Saved] fig3_spectral_radius")

def plot_fig4_lyapunov(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.0))
    
    x = np.array(data["horizon"])
    kappas = ["kappa_1", "kappa_5", "kappa_20", "kappa_50"]
    labels = [r"$\kappa(\mathbf{P})=1$", r"$\kappa(\mathbf{P})=5$", r"$\kappa(\mathbf{P})=20$", r"$\kappa(\mathbf{P})=50$"]
    colors = [COLORS["indigo"], COLORS["teal"], COLORS["amber"], COLORS["purple"]]
    
    # Subplot (a): Energy curves with full visibility across all 4 kappas + unregularized
    for k_key, label, col in zip(kappas, labels, colors):
        geo_v = np.array(data["lyapunov_energy_by_kappa"][k_key]["geometric"])
        ax1.plot(x, geo_v, label=f"Geo ({label})", color=col, lw=1.8)
        
    unreg_v = np.array(data["lyapunov_energy_by_kappa"]["kappa_20"]["unregularized"])
    ax1.plot(x, unreg_v, label=r"Unreg ($\kappa=20$)", color=COLORS["crimson"], lw=2.0, linestyle="--")
    
    ax1.set_xlabel("Execution Horizon $t$ (Steps)", fontsize=9.5)
    ax1.set_ylabel(r"Lyapunov Energy $V(\mathbf{h}_t) = \mathbf{h}_t^\top \mathbf{P} \mathbf{h}_t$", fontsize=9.5)
    ax1.set_title(r"(a) Lyapunov Energy Trajectories", fontsize=10, pad=8)
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 64)
    ax1.grid(True, linestyle=":", alpha=0.6)
    # Headroom legend in top-left: generous headroom ensuring curves stay completely below
    ax1.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=True, framealpha=0.92)
    
    # Subplot (b): 2D Phase Portrait
    unreg_pts = np.array(data["phase_space"]["unregularized"])
    geo_pts = np.array(data["phase_space"]["geometric"])
    
    ax2.plot(unreg_pts[:, 0], unreg_pts[:, 1], color=COLORS["crimson"], lw=1.4, alpha=0.85, label="Runaway Escape")
    ax2.plot(geo_pts[:, 0], geo_pts[:, 1], color=COLORS["teal"], lw=2.0, label="Attractor Basin")
    ax2.scatter([geo_pts[0, 0]], [geo_pts[0, 1]], color=COLORS["indigo"], s=45, zorder=5, label=r"Initial State $\mathbf{h}_0$")
    ax2.scatter([0], [0], color="#0F172A", marker="x", s=55, zorder=5, lw=2, label="Attractor Origin")
    
    ax2.set_xlabel(r"State Coordinate $h_{t,1}$", fontsize=9.5)
    ax2.set_ylabel(r"State Coordinate $h_{t,2}$", fontsize=9.5)
    ax2.set_title(r"(b) Phase Portrait $(\mathbf{h}_{t,1}, \mathbf{h}_{t,2})$", fontsize=10, pad=8)
    ax2.set_xlim(-0.35, 0.35)
    ax2.set_ylim(-0.38, 0.32)
    ax2.grid(True, linestyle=":", alpha=0.6)
    # Legend in upper right of phase portrait where trajectory is completely empty
    ax2.legend(loc="upper right", fontsize=7.2, frameon=True, framealpha=0.92)
    
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig4_lyapunov.pdf"))
        fig.savefig(os.path.join(d, "fig4_lyapunov.png"))
    plt.close(fig)
    print("  [Saved] fig4_lyapunov")

def plot_fig5_attention(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    tokens = np.array(data["token_indices"])
    delta = data["delta"]
    
    color_map = {
        "10": COLORS["indigo"],
        "30": COLORS["teal"],
        "60": COLORS["amber"],
        "90": COLORS["crimson"]
    }
    
    for step_str in ["10", "30", "60", "90"]:
        w = np.array(data["profiles"][step_str])
        # Only plot up to token index <= step
        step_val = int(step_str)
        ax.plot(tokens[:step_val], w[:step_val], label=f"$t = {step_str}$", color=color_map[step_str], lw=2.0)
        
    # Fidelity threshold
    ax.axhline(y=delta, color="#334155", linestyle="--", lw=1.4, label=r"Fidelity Threshold $\delta = 0.01$")
    
    ax.set_xlabel("Historical Token Index $i$")
    ax.set_ylabel(r"Normalized Attention Weight $\alpha_i^{(t)}$")
    ax.set_title(r"Context Saturation Profile and Attention Weight Decay over Horizons")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 0.16)
    ax.grid(True)
    ax.legend(loc="upper right")
    
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig5_attention.pdf"))
        fig.savefig(os.path.join(d, "fig5_attention.png"))
    plt.close(fig)
    print("  [Saved] fig5_attention")

def plot_fig6_eigenvalue_spectra(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 4.0), sharey=True)
    theta = np.linspace(0, 2*np.pi, 200)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
    
    # Subplot 1: Prompt Wrapper (Breaches unit circle)
    h_unit, = ax1.plot(unit_circle_x, unit_circle_y, color="#334155", linestyle="--", lw=1.4, label=r"Unit Circle ($|\lambda|=1$)")
    snapshots = ["10", "50", "90"]
    colors = [COLORS["teal"], COLORS["amber"], COLORS["crimson"]]
    h_scatters = []
    
    for s_step, col in zip(snapshots, colors):
        real = data["eigenvalues"]["prompt_wrapper"][s_step]["real"]
        imag = data["eigenvalues"]["prompt_wrapper"][s_step]["imag"]
        sc = ax1.scatter(real, imag, s=16, color=col, alpha=0.75, label=f"$t = {s_step}$")
        h_scatters.append(sc)
        
    ax1.set_title(r"(a) Prompt Wrapper: Spectral Outward Drift", fontsize=10, pad=8)
    ax1.set_xlabel(r"$\mathrm{Re}(\lambda)$", fontsize=9.5)
    ax1.set_ylabel(r"$\mathrm{Im}(\lambda)$", fontsize=9.5)
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Subplot 2: Geometric Persistence (Strictly bounded)
    ax2.plot(unit_circle_x, unit_circle_y, color="#334155", linestyle="--", lw=1.4)
    h_bound, = ax2.plot(0.82 * unit_circle_x, 0.82 * unit_circle_y, color=COLORS["indigo"], linestyle=":", lw=1.6, label=r"Contractive Bound ($0.82$)")
    
    for s_step, col in zip(snapshots, colors):
        real = data["eigenvalues"]["geometric"][s_step]["real"]
        imag = data["eigenvalues"]["geometric"][s_step]["imag"]
        ax2.scatter(real, imag, s=16, color=col, alpha=0.75)
        
    ax2.set_title(r"(b) Geometric Persistence: Contractive Invariance", fontsize=10, pad=8)
    ax2.set_xlabel(r"$\mathrm{Re}(\lambda)$", fontsize=9.5)
    ax2.set_xlim(-1.4, 1.4)
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # Unified Legend at top of figure
    handles = [h_unit, h_bound, h_scatters[0], h_scatters[1], h_scatters[2]]
    labels = [r"Unit Circle ($|\lambda|=1$)", r"Contractive Bound ($0.82$)", r"$t = 10$", r"$t = 50$", r"$t = 90$"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5, fontsize=8, frameon=True, framealpha=0.92)
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig6_eigenvalue_spectra.pdf"))
        fig.savefig(os.path.join(d, "fig6_eigenvalue_spectra.png"))
    plt.close(fig)
    print("  [Saved] fig6_eigenvalue_spectra")

def plot_fig7_constraint_surface(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig = plt.figure(figsize=(7.6, 4.0))
    ax = fig.add_subplot(111, projection='3d')
    
    dims = np.array(data["surface_grid"]["dims"])
    kappas = np.array(data["surface_grid"]["kappas"])
    Z = np.array(data["surface_grid"]["rhs_matrix"])
    
    D, K = np.meshgrid(dims, kappas)
    
    surf = ax.plot_surface(D, K, Z.T, cmap="viridis", edgecolor='none', alpha=0.88)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.08)
    cbar.set_label(r"Admissible Error Upper Bound $\frac{\delta_{\mathrm{stable}}}{\sqrt{d}\kappa(\mathbf{P})}$", fontsize=8.5)
    
    ax.set_xlabel("Vector Dimension $d$", labelpad=6)
    ax.set_ylabel(r"Condition Number $\kappa(\mathbf{P})$", labelpad=6)
    ax.set_zlabel(r"Bound Value", labelpad=6)
    ax.set_title(r"Agentic Constraint Boundary (Theorem 3): Shrinking Stability Volume", pad=12)
    ax.view_init(elev=28, azim=-55)
    
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig7_constraint_surface.pdf"))
        fig.savefig(os.path.join(d, "fig7_constraint_surface.png"))
    plt.close(fig)
    print("  [Saved] fig7_constraint_surface")

def plot_fig8_grounding_sensitivity(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.6, 5.0), sharex=True, sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    gap_types = ["nominal", "drift", "pareto_heavy", "adversarial"]
    titles = ["(a) Nominal Gaussian Error", "(b) Contextual Drift Error", "(c) Heavy-Tailed Pareto Bursts", "(d) Adversarial Oscillatory Gap"]
    
    x = np.arange(1, 101)
    
    for ax, g_type, title in zip(axes, gap_types, titles):
        geo = np.array(data[g_type]["geometric_mean_dist"])
        prompt = np.array(data[g_type]["prompt_mean_dist"])
        unreg = np.array(data[g_type]["unreg_mean_dist"])
        
        ax.plot(x, geo, label="Geometric (Ours)", color=COLORS["indigo"], lw=1.8)
        ax.plot(x, prompt, label="Prompt Wrapper", color=COLORS["crimson"], lw=1.6)
        ax.plot(x, unreg, label="Unregularized", color=COLORS["amber"], lw=1.5, linestyle="--")
        
        # Divergence threshold
        ax.axhline(y=1.5, color=COLORS["slate"], linestyle=":", lw=1.2)
        ax.set_title(title, fontsize=9.5)
        ax.grid(True)
        
    ax1.set_ylabel("Distance to Goal")
    ax3.set_ylabel("Distance to Goal")
    ax3.set_xlabel("Horizon Step $t$")
    ax4.set_xlabel("Horizon Step $t$")
    ax1.set_ylim(0, 2.2)
    ax1.legend(loc="upper left", fontsize=7.5)
    
    fig.suptitle("Orchestration State Fidelity Degradation Across Action Grounding Regimes", fontsize=11, y=0.98)
    
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig8_grounding_sensitivity.pdf"))
        fig.savefig(os.path.join(d, "fig8_grounding_sensitivity.png"))
    plt.close(fig)
    print("  [Saved] fig8_grounding_sensitivity")

def plot_fig9_ablation_multi_metric(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.0))
    
    # Subplot 1: Success rate curves across horizons T in [10, 150]
    horizons = data["horizons"]
    rates = data["table2_success_rates"]
    
    palette = {
        "Geometric (Ours)": (COLORS["indigo"], "-"),
        "Prompt Wrapper": (COLORS["crimson"], "-"),
        "External Memory (DNC)": (COLORS["purple"], "-."),
        "Sliding Window (W=16)": (COLORS["slate"], ":"),
        "Unregularized": (COLORS["amber"], "--")
    }
    
    for name, (col, style) in palette.items():
        data_key = "Geometric Persistence (Ours)" if name == "Geometric (Ours)" else name
        curve = []
        for h in horizons:
            v = rates[data_key][str(h)]
            curve.append(v["rate"] if isinstance(v, dict) else v)
        ax1.plot(horizons, curve, label=name, color=col, linestyle=style, lw=2.0, marker='o', markersize=4)
        
    ax1.set_xlabel("Execution Horizon $T$ (Steps)", fontsize=9.5)
    ax1.set_ylabel("Trajectory-Boundedness Rate (%)", fontsize=9.5)
    ax1.set_title("(a) Long-Horizon Boundedness Rate Scaling", fontsize=10, pad=6)
    ax1.set_xlim(10, 150)
    # Headroom to 132 ensures curves (<=100%) never intersect legend
    ax1.set_ylim(-2, 132)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", ncol=2, fontsize=7.2, frameon=True, framealpha=0.92)
    
    # Subplot 2: Grouped bar chart of stability metrics at T=100
    metrics = data["table3_stability_metrics"]
    primary_agents = ["Unregularized", "Prompt Wrapper", "Sliding Window (W=16)", "External Memory (DNC)", "Geometric Persistence (Ours)"]
    short_names = ["Unreg", "Prompt", "Sliding", "DNC", "Geo (Ours)"]
    
    mttf = [metrics[a]["mean_ttf"] for a in primary_agents]
    div_rate = [100.0 - metrics[a]["success_rate"] for a in primary_agents]
    
    bar_x = np.arange(len(primary_agents))
    width = 0.36
    
    ax2.bar(bar_x - width/2, mttf, width, label="Mean Time to Failure (Steps)", color=COLORS["teal"], alpha=0.9)
    ax2.bar(bar_x + width/2, div_rate, width, label="State Divergence Rate (%)", color=COLORS["crimson"], alpha=0.9)
    
    ax2.set_xticks(bar_x)
    ax2.set_xticklabels(short_names, rotation=20, ha="right", fontsize=8.5)
    ax2.set_xlabel("Architectural Paradigm", fontsize=9.5)
    ax2.set_ylabel("Metric Value", fontsize=9.5)
    ax2.set_title("(b) Stability Diagnostic Benchmark ($T=100$)", fontsize=10, pad=6)
    ax2.set_ylim(0, 130)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", ncol=1, fontsize=7.5, frameon=True, framealpha=0.92)
    
    fig.tight_layout()
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig9_ablation_multi_metric.pdf"))
        fig.savefig(os.path.join(d, "fig9_ablation_multi_metric.png"))
    plt.close(fig)
    print("  [Saved] fig9_ablation_multi_metric")

def plot_fig10_attention_heatmap(data_path: str, out_dirs: list):
    with open(data_path, "r") as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    heatmap = np.array(data["heatmap_2d"])
    
    im = ax.imshow(heatmap, cmap="magma", aspect="auto", origin="lower",
                   extent=[1, 100, 1, 100], vmin=0, vmax=0.06)
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(r"Attention Weight $\alpha_i^{(t)}$", fontsize=9.5)
    
    # Overlay theoretical critical horizon line T* = 60
    ax.axhline(y=60, color=COLORS["crimson"], linestyle="--", lw=1.8, label=r"Critical Horizon Boundary $T^* = 60$")
    ax.plot([1, 100], [1, 100], color="#F8FAFC", linestyle=":", lw=1.2, alpha=0.8, label=r"Causal Diagonal $i = t$")
    
    ax.set_xlabel("Historical Context Token Index $i$")
    ax.set_ylabel("Execution Time Step $t$")
    ax.set_title(r"Spatiotemporal Attention Saturation Landscape: The Context Erosion Frontier")
    ax.legend(loc="upper left", facecolor="#1E293B", labelcolor="white", edgecolor="none", fontsize=8.5)
    
    for d in out_dirs:
        fig.savefig(os.path.join(d, "fig10_attention_heatmap.pdf"))
        fig.savefig(os.path.join(d, "fig10_attention_heatmap.png"))
    plt.close(fig)
    print("  [Saved] fig10_attention_heatmap")

def generate_all():
    print("=" * 60)
    print("GENERATING BENCHMARK FIGURES (PDF & PNG)")
    print("=" * 60)
    
    vis_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(vis_dir)
    
    if os.path.exists(os.path.join(parent_dir, "results")):
        res_dir = os.path.join(parent_dir, "results")
    elif os.path.exists("results"):
        res_dir = "results"
    else:
        res_dir = "Implementation/results"
        
    if os.path.exists(os.path.join(parent_dir, "figures")):
        fig_dir = os.path.join(parent_dir, "figures")
    elif os.path.exists("figures"):
        fig_dir = "figures"
    else:
        fig_dir = "Implementation/figures"
        
    os.makedirs(fig_dir, exist_ok=True)
    out_dirs = [fig_dir]
    
    # Only copy to parent workspace if main.tex exists (local paper build mode)
    paper_dir = os.path.dirname(parent_dir)
    if os.path.exists(os.path.join(paper_dir, "main.tex")):
        out_dirs.append(paper_dir)
    elif os.path.exists("main.tex"):
        out_dirs.append(".")
    
    plot_fig2_error_norm(os.path.join(res_dir, "exp1_error_accumulation.json"), out_dirs)
    plot_fig3_spectral_radius(os.path.join(res_dir, "exp2_spectral_analysis.json"), out_dirs)
    plot_fig4_lyapunov(os.path.join(res_dir, "exp3_lyapunov_stability.json"), out_dirs)
    plot_fig5_attention(os.path.join(res_dir, "exp4_attention_saturation.json"), out_dirs)
    plot_fig6_eigenvalue_spectra(os.path.join(res_dir, "exp2_spectral_analysis.json"), out_dirs)
    plot_fig7_constraint_surface(os.path.join(res_dir, "exp5_constraint_equation.json"), out_dirs)
    plot_fig8_grounding_sensitivity(os.path.join(res_dir, "exp6_grounding_gap_drift.json"), out_dirs)
    plot_fig9_ablation_multi_metric(os.path.join(res_dir, "exp7_ablation_benchmarks.json"), out_dirs)
    plot_fig10_attention_heatmap(os.path.join(res_dir, "exp4_attention_saturation.json"), out_dirs)
    
    print("=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    generate_all()
