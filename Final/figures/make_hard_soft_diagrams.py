"""Generates Fig. Hard-Routing and Fig. Soft-MoE actor architecture diagrams,
matching the visual style of learned_router_architecture.png. Dimensions pulled
directly from 3_Agent_Select_1/Finrlmain.py (class Actor) and
3_Agent_Select_3/agents_moe.py (class ActorMoE).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

BOX_FC = "#eef3fb"
BOX_EC = "#2c3e50"
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
# Figure — Hard Routing actor
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 8.2)
ax.axis("off")
ax.set_title("Hard Routing Actor Architecture", fontsize=14, fontweight="bold", pad=18)
ax.text(5.5, 7.95, "Regime label selects exactly one of 4 specialist heads;\n"
                    "only that head's parameters receive gradient on this step",
        ha="center", fontsize=9.5, style="italic", color="#555")

box(ax, 0.3, 5.5, 2.7, 1.1, "Observation  x",
    "dim = 1597\n(1596 market state +\n1 scalar regime label)", fontsize=10)

box(ax, 0.3, 3.3, 2.7, 1.0, "Feature Extractor",
    "Linear(1597→256) → ReLU\nLinear(256→256) → ReLU\n(full x, label included)",
    fc="#eaf1fb", ec="#1d4e89", fontsize=9.8)

box(ax, 0.3, 1.6, 2.7, 1.0, "x[:, -1]",
    "regime label\n(int 0-3)", fc=GATE_FC, ec=GATE_EC, fontsize=10)

box(ax, 3.9, 0.6, 2.4, 1.0, "Mask / Select",
    "argmax-style hard\nselection of 1 of 4 heads", fc=GATE_FC, ec=GATE_EC, fontsize=9.8)

head_labels = ["Head 0\n(Bear)", "Head 1\n(Sideways ↓)", "Head 2\n(Sideways ↑)", "Head 3\n(Bull)"]
head_colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"]
head_y = [7.0 - i * 1.05 for i in range(4)]
for lab, c, hy in zip(head_labels, head_colors, head_y):
    box(ax, 6.7, hy, 2.0, 0.85, lab, "Linear(256→38)", fc=c, ec="#333", fontsize=8.7)

box(ax, 9.1, 3.2, 1.7, 1.1, "mean = 0.1 × raw",
    "+ clamp(std,\n1e-3, 1)", fontsize=8.8)
box(ax, 9.1, 1.7, 1.7, 1.1, "Normal\n(mean, std)",
    "sampled weights\ndim = 38", fontsize=8.8)

# arrows
arrow(ax, (1.65, 5.5), (1.65, 4.3))
arrow(ax, (1.65, 3.3), (1.65, 2.6))
arrow(ax, (3.0, 2.1), (3.9, 1.1))
for hy in head_y:
    arrow(ax, (3.0, 3.8), (6.7, hy + 0.42), lw=1.0, color="#888")
# Mask/Select's choice of head, routed up to the head column it controls
elbow_arrow(ax, [(5.1, 1.6), (5.1, 5.6), (6.7, 5.6)], lw=1.2, color=GATE_EC)
for hy in head_y:
    arrow(ax, (8.7, hy + 0.42), (9.1, 3.75), lw=1.0, color="#888")
arrow(ax, (9.95, 3.2), (9.95, 2.8))

plt.tight_layout()
fig.savefig("hard_rarl_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig("hard_rarl_architecture.pdf", bbox_inches="tight")
plt.close(fig)


# ----------------------------------------------------------------------------
# Figure — Soft MoE (Soft RA-RL) actor
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.1, 8.2))
ax.set_xlim(0, 12.1)
ax.set_ylim(0, 8.2)
ax.axis("off")
ax.set_title("Soft MoE (Soft RA-RL) Actor Architecture", fontsize=14, fontweight="bold", pad=18)
ax.text(5.75, 7.95, "All 4 heads evaluated every step; blended directly by the\n"
                     "HMM posterior — no separate learned gating network",
        ha="center", fontsize=9.5, style="italic", color="#555")

box(ax, 0.2, 5.9, 2.6, 1.0, "Observation  x",
    "dim = 1600\n(1596 market state +\n4 HMM regime probs)", fontsize=10)

box(ax, 0.2, 3.9, 2.6, 0.9, "x[:, :-4]",
    "market-state slice\ndim = 1596", fontsize=10)
box(ax, 0.2, 2.7, 2.6, 0.9, "x[:, -4:]",
    "HMM regime probs\ndim = 4", fc=GATE_FC, ec=GATE_EC, fontsize=10)

box(ax, 3.4, 4.5, 2.7, 1.9, "Feature Extractor",
    "Linear(1596→256) → ReLU\nLinear(256→256) → ReLU",
    fc="#eaf1fb", ec="#1d4e89", fontsize=10.3)

exp_labels = ["Expert 0\n(Bear)", "Expert 1\n(Sideways ↓)", "Expert 2\n(Sideways ↑)", "Expert 3\n(Bull)"]
exp_colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"]
exp_y = [6.5 - i * 1.05 for i in range(4)]
for lab, c, ey in zip(exp_labels, exp_colors, exp_y):
    box(ax, 6.8, ey, 2.0, 0.85, lab, "Linear(256→38)", fc=c, ec="#333", fontsize=8.7)

box(ax, 9.3, 4.0, 2.0, 1.3, "Σ pᵢ · expertᵢ",
    "weighted sum,\nweights = HMM\nposterior, dim = 38", fontsize=9.3)
box(ax, 9.3, 2.5, 2.0, 1.1, "mean = 0.1 × logits",
    "+ clamp(std, 1e-3, 1)", fontsize=9.0)
box(ax, 9.3, 1.0, 2.0, 1.1, "Normal(mean, std)",
    "sampled portfolio\nweights, dim = 38", fontsize=9.0)

arrow(ax, (1.5, 5.9), (1.5, 4.8))
arrow(ax, (1.5, 3.9), (1.5, 3.6))
arrow(ax, (2.8, 4.35), (3.4, 5.2))
for ey in exp_y:
    arrow(ax, (6.1, 5.4), (6.8, ey + 0.42), lw=1.0, color="#888")
for ey in exp_y:
    arrow(ax, (8.8, ey + 0.42), (9.3, 4.6), lw=1.0, color="#888")
# regime probs feed directly into the weighted-sum block as blend weights,
# routed below the expert column and up the outside-right margin to avoid
# cutting through Expert 3 (Bull) and the mean/Normal boxes below the sum box
elbow_arrow(ax, [(2.8, 3.15), (2.8, 0.3), (11.75, 0.3), (11.75, 4.65), (11.3, 4.65)],
            lw=1.2, color=GATE_EC)
arrow(ax, (10.3, 4.0), (10.3, 3.6))
arrow(ax, (10.3, 2.5), (10.3, 2.1))

plt.tight_layout()
fig.savefig("soft_rarl_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig("soft_rarl_architecture.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved hard_rarl_architecture.{png,pdf} and soft_rarl_architecture.{png,pdf}")
