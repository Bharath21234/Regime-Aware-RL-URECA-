"""Generates Fig. HMM-architecture and Fig. Learned-Router-architecture for the
ICAIF paper. Dimensions are pulled directly from 3_Agent_Select_4 (hmm_probabilistic.py,
agents_router.py, env_router.py, main_router.py) so the figure matches the code exactly.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

plt.rcParams["font.family"] = "DejaVu Sans"

BOX_FC = "#eef3fb"
BOX_EC = "#2c3e50"
HEAD_FC = "#dfeee3"
HEAD_EC = "#1e7a3a"
GATE_FC = "#fdf0e0"
GATE_EC = "#b5651d"
TXT = "#1a1a1a"


def box(ax, x, y, w, h, title, subtitle=None, fc=BOX_FC, ec=BOX_EC, fontsize=11, title_weight="bold"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.6, edgecolor=ec, facecolor=fc, mutation_aspect=1)
    ax.add_patch(b)
    cx, cy = x + w / 2, y + h / 2
    if subtitle:
        ax.text(cx, cy + h * 0.14, title, ha="center", va="center", fontsize=fontsize,
                fontweight=title_weight, color=TXT)
        ax.text(cx, cy - h * 0.20, subtitle, ha="center", va="center", fontsize=fontsize - 2.5,
                color="#444", linespacing=1.5)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fontsize,
                fontweight=title_weight, color=TXT)
    return b


def arrow(ax, p1, p2, lw=1.6, color="#444", style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def elbow_arrow(ax, points, lw=1.2, color="#444"):
    """Draws straight segments through waypoints, with an arrowhead only on
    the final segment. Used to route a connection around boxes that sit
    between the source and destination."""
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]],
                color=color, lw=lw, solid_capstyle="round", zorder=1)
    arrow(ax, points[-2], points[-1], lw=lw, color=color)


# ----------------------------------------------------------------------------
# Figure 1 — HMM regime-detection pipeline
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.2)
ax.axis("off")
ax.set_title("Regime Detection Pipeline (Gaussian HMM)", fontsize=14, fontweight="bold", pad=14)

box(ax, 0.3, 4.6, 2.5, 1.0, "Raw OHLCV Panel",
    "38 tickers + macro ETFs\ndaily close prices", fontsize=10.5)

box(ax, 0.3, 2.9, 2.5, 1.0, "Feature Engineering",
    "20-day rolling window:\nmean return, volatility,\nmomentum, avg. corr.", fontsize=10.5)

box(ax, 0.3, 1.2, 2.5, 1.0, "Standardisation",
    "z-score per feature\n(train-set mean/std)", fontsize=10.5)

box(ax, 3.7, 2.05, 3.0, 2.4, "Gaussian HMM",
    "GaussianHMM(n_components=4,\ncovariance_type='full',\nn_iter=1000)\n\nEM fit on 4-dim feature\nsequence  X ∈ ℝ^(T×4)",
    fc="#eaf1fb", ec="#1d4e89", fontsize=11)

box(ax, 7.6, 4.4, 3.1, 1.3, "Regime Ordering",
    "sort states by\n(momentum − volatility)\nargsort → Bear...Bull", fc=GATE_FC, ec=GATE_EC, fontsize=10.5)

box(ax, 7.6, 2.3, 3.1, 1.5, "predict_proba(·)",
    "posterior regime\nprobabilities per day\np ∈ Δ³ (4-simplex)", fontsize=10.5)

# 4 regime probability outputs
labels = ["Bear", "Sideways ↓", "Sideways ↑", "Bull"]
colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"]
y0 = 0.15
for i, (lab, c) in enumerate(zip(labels, colors)):
    bx = 7.4 + i * 0.95
    box(ax, bx, y0, 0.85, 0.55, f"regime_p_{i}\n{lab}", fc=c, ec="#333", fontsize=7.3, title_weight="normal")

# arrows
arrow(ax, (1.55, 4.6), (1.55, 3.9))
arrow(ax, (1.55, 2.9), (1.55, 2.2))
arrow(ax, (2.8, 1.7), (3.7, 2.9))
arrow(ax, (6.7, 3.6), (7.6, 4.6))
arrow(ax, (6.7, 3.0), (7.6, 3.0))
arrow(ax, (9.15, 4.4), (9.15, 3.8))
for i in range(4):
    arrow(ax, (8.0 + i * 0.4, 2.3), (7.8 + i * 0.95, 0.70))

ax.text(5.45, 5.45, "fit() — done once, offline, on macro ETF panel\n(separate from RL training)",
        ha="center", fontsize=9.5, style="italic", color="#555")
ax.text(5.45, 0.9, "↓ concatenated onto RL\nobservation as last 4 dims\nof state vector (see Fig. 2)",
        ha="center", fontsize=9, style="italic", color="#555")

plt.tight_layout()
fig.savefig("hmm_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig("hmm_architecture.pdf", bbox_inches="tight")
plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2 — Learned Router actor architecture
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 8.8))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 8.8)
ax.axis("off")
ax.set_title("Learned-Router Actor Architecture", fontsize=14, fontweight="bold", pad=18)

# Input
box(ax, 0.2, 5.9, 2.6, 1.0, "Observation  x",
    "dim = 1600\n(1596 market state +\n4 HMM regime probs)", fontsize=10)

box(ax, 0.2, 3.9, 2.6, 0.9, "x[:, :-4]",
    "market-state slice\ndim = 1596", fontsize=10)
box(ax, 0.2, 2.7, 2.6, 0.9, "x[:, -4:]",
    "HMM regime probs\ndim = 4", fc=GATE_FC, ec=GATE_EC, fontsize=10)

# Feature extractor
box(ax, 3.4, 4.5, 2.9, 1.9, "Feature Extractor",
    "Linear(1596→256) → ReLU\nLinear(256→256) → ReLU\n\nshared trunk, regime-blind",
    fc="#eaf1fb", ec="#1d4e89", fontsize=10.3)

# Router MLP
box(ax, 3.4, 1.6, 2.9, 2.1, "Router MLP",
    "concat(features, regime_p)\ndim = 260\n\nLinear(260→64) → ReLU\nLinear(64→4) → Softmax",
    fc=GATE_FC, ec=GATE_EC, fontsize=10.3)
box(ax, 3.95, 0.25, 1.8, 0.85, "routing_weights",
    "w ∈ Δ³  (dim 4)", fc=GATE_FC, ec=GATE_EC, fontsize=9.3)

# 4 Expert heads
exp_labels = ["Expert 0\n(Bear)", "Expert 1\n(Sideways ↓)", "Expert 2\n(Sideways ↑)", "Expert 3\n(Bull)"]
exp_colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"]
exp_y = [6.5 - i * 1.05 for i in range(4)]
for lab, c, by in zip(exp_labels, exp_colors, exp_y):
    box(ax, 7.0, by, 2.0, 0.85, lab, "Linear(256→38)", fc=c, ec="#333", fontsize=8.7)

# Combine
box(ax, 9.5, 4.0, 2.2, 1.3, "Σ wᵢ · expertᵢ",
    "weighted sum over\nexpert logits\ndim = 38", fontsize=9.8)

box(ax, 9.5, 2.5, 2.2, 1.1, "mean = 0.1 × logits",
    "+ clamp(std, 1e-3, 1)", fontsize=9.3)

box(ax, 9.5, 1.0, 2.2, 1.1, "Normal(mean, std)",
    "sampled portfolio\nweights, dim = 38", fc="#eef3fb", ec="#2c3e50", fontsize=9.5)

# Critic (small inset)
box(ax, 0.2, 0.25, 2.6, 1.1, "Critic V(x)",
    "Linear(1600→256→256→1)\n(separate network, full x)", fc="#f3f0fa", ec="#5b3a8c", fontsize=9.3)

# arrows
arrow(ax, (1.5, 5.9), (1.5, 4.8))
arrow(ax, (1.5, 3.9), (1.5, 3.6))
arrow(ax, (2.8, 4.35), (3.4, 5.2))
arrow(ax, (2.8, 3.15), (3.4, 2.6))
arrow(ax, (6.3, 4.5), (6.3, 3.7))   # features -> router input
arrow(ax, (5.85, 1.6), (4.85, 1.1))  # router -> routing_weights box
# feature extractor -> each expert head
for by in exp_y:
    arrow(ax, (6.3, 5.4), (7.0, by + 0.42), lw=1.0, color="#888")
# each expert head -> weighted-sum box
for by in exp_y:
    arrow(ax, (9.0, by + 0.42), (9.5, 4.65), lw=1.0, color="#888")
# routing_weights -> weighted-sum box, routed below the expert column and up
# the outside-right margin to avoid cutting through Expert 3 (Bull) and the
# mean/Normal boxes stacked below the weighted-sum box
elbow_arrow(ax, [(5.75, 0.65), (5.75, 0.12), (12.15, 0.12), (12.15, 4.65), (11.7, 4.65)],
            lw=1.2, color=GATE_EC)
arrow(ax, (10.6, 4.0), (10.6, 3.6))
arrow(ax, (10.6, 2.5), (10.6, 2.1))
arrow(ax, (3.95, 0.7), (2.8, 0.8), lw=1.2, color="#5b3a8c", style="-|>")

ax.text(6.0, 8.3, "Soft blend over 4 regime-specialised heads, gated by a learned\n"
                  "router conditioned on BOTH state features and the HMM signal",
        ha="center", fontsize=9.5, style="italic", color="#555")

plt.tight_layout()
fig.savefig("learned_router_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig("learned_router_architecture.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved hmm_architecture.{png,pdf} and learned_router_architecture.{png,pdf}")
