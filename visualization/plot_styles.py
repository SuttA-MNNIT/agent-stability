"""
IEEE Journal Publication Plot Styling Configuration.
Provides consistent colors, typography, line weights, and grid settings.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Set IEEE Journal standard style parameters
def apply_ieee_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Utopia", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CBD5E1",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#334155",
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.autolayout": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

# Curated Palette
COLORS = {
    "indigo": "#1E40AF",       # Proposed Geometric Persistence
    "teal": "#0D9488",         # Attractor / Stable
    "crimson": "#B91C1C",      # Divergent / Unregularized / Noise
    "amber": "#D97706",        # Prompt Wrapper / Warning
    "purple": "#6D28D9",       # DNC Memory Bank
    "slate": "#475569",        # Sliding Window
    "emerald": "#047857",      # Bounds
    "gray": "#94A3B8"
}
