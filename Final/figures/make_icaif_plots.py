"""ICAIF '26 figures (corrected batchfix mechanics — results_log sections 5a/8a).

  fig 1 - seed_sharpe_dots.png : per-seed Sharpe dot plot, A2C (corrected)
          vs PPO panels. THE narrative figure: soft's stability, hard's
          2/3-seed collapse under A2C, PPO rescuing hard, router collapse.
  fig 2 - bar_corrected_a2c.png : Table 1 bar chart (n=3, corrected A2C).
  fig 3 - equity_curve_corrected_seed1.png : cumulative-return crop from the
          corrected run's server-rendered metrics_over_time PNGs (seed 1).

The URECA scripts (make_results_plots.py, make_equity_curve.py) plot the
OLD-mechanics numbers and are kept unchanged for the URECA paper record.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

plt.rcParams["font.family"] = "DejaVu Sans"

HARD_C = "#c0392b"
SOFT_C = "#1d8a4e"
ROUTER_C = "#7f7f7f"

# Per-seed Sharpe, corrected A2C single-period (results_log 8a) and PPO (5a).
# No corrected-mechanics A2C Router run exists; Router appears in the PPO
# panel only (its A2C collapse under original mechanics is cited in text).
a2c = {"Hard RA-RL": [-0.0825, -0.0526, 0.5626],
       "Soft RA-RL": [0.2616, 0.7349, 0.3897]}
ppo = {"Hard RA-RL": [0.886, 0.809, 0.483],
       "Soft RA-RL": [0.538, 0.975, 0.808],
       "Learned Router": [0.454, 0.767, -0.281]}
colors = {"Hard RA-RL": HARD_C, "Soft RA-RL": SOFT_C, "Learned Router": ROUTER_C}

# ---------------------------------------------------------------- fig 1
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True,
                         gridspec_kw={"width_ratios": [2, 3]})
for ax, data, title in zip(axes, [a2c, ppo],
                           ["A2C (no trust region)", "PPO (clipped objective)"]):
    for i, (name, seeds) in enumerate(data.items()):
        x = np.full(len(seeds), i)
        ax.scatter(x, seeds, s=70, color=colors[name], zorder=3,
                   edgecolor="#333", linewidth=0.8)
        ax.hlines(np.mean(seeds), i - 0.28, i + 0.28, color=colors[name],
                  linewidth=2.5, zorder=2)
    ax.axhline(0, color="#999", lw=0.9, linestyle="--", zorder=1)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in data], fontsize=9)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlim(-0.6, len(data) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Annualised Sharpe (test period)")
fig.suptitle("Per-seed Sharpe, single-period test (2022–2024), 3 shared seeds; bars = means",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("seed_sharpe_dots.png", dpi=300, bbox_inches="tight")
fig.savefig("seed_sharpe_dots.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2
metrics = ["Return (%)", "Sharpe", "Max Drawdown (%)", "Sortino"]
hard_mean = [0.28, 0.142, 48.87, 0.144]
hard_std = [28.00, 0.364, 3.93, 0.364]
soft_mean = [26.24, 0.462, 57.23, 0.469]
soft_std = [29.83, 0.245, 2.86, 0.242]

fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
for ax, m, hm, hs, sm, ss in zip(axes, metrics, hard_mean, hard_std, soft_mean, soft_std):
    ax.bar([0, 1], [hm, sm], yerr=[hs, ss], capsize=5, color=[HARD_C, SOFT_C],
           edgecolor="#333", width=0.6, error_kw={"elinewidth": 1.3})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hard\nRA-RL", "Soft\nRA-RL"], fontsize=10)
    ax.set_title(m, fontsize=11, fontweight="bold")
    ax.axhline(0, color="#999", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Corrected mechanics, A2C, single-period (n=3 seeds, mean ± std)",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig("bar_corrected_a2c.png", dpi=300, bbox_inches="tight")
fig.savefig("bar_corrected_a2c.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 3
CROP_BOX = (0, 75, 2084, 640)  # same crop as make_equity_curve.py
hard_img = Image.open("../results/hard_bf/seed_1_metrics_over_time.png").crop(CROP_BOX)
soft_img = Image.open("../results/moe_bf/seed_1_metrics_over_time.png").crop(CROP_BOX)

w, h = hard_img.size
label_h, gap = 50, 12
canvas = Image.new("RGB", (w, label_h * 2 + h * 2 + gap), "white")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
except Exception:
    font = ImageFont.load_default()

draw.text((10, 10), "Hard RA-RL — Seed 1 (Return -14.2%, Sharpe -0.05)",
          fill="#1a1a1a", font=font)
canvas.paste(hard_img, (0, label_h))
draw.text((10, label_h + h + gap + 10),
          "Soft RA-RL — Seed 1 (Return +59.9%, Sharpe +0.73)",
          fill="#1a1a1a", font=font)
canvas.paste(soft_img, (0, label_h * 2 + h + gap))
canvas.save("equity_curve_corrected_seed1.png", dpi=(300, 300))

# ---------------------------------------------------------------- fig 4
# Corrected walk-forward per-window Sharpe (results_log section 10b),
# read from the run's own per_window_summary.json. Replaces the old
# walkforward_sharpe.png (superseded mechanics — do not reuse).
import json, os
wf_dir = os.path.join(os.path.dirname(__file__), "..",
                      "Walkforward", "results_batchfix_2seed")
rows = json.load(open(os.path.join(wf_dir, "per_window_summary.json")))
by = {}
for r in rows:
    by.setdefault(r["method"], {})[r["window"]] = r["Sharpe"]

windows = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
wlabels = ["W1\nCOVID", "W2\nRecov.", "W3\nBull", "W4\nBull",
           "W5\nBear\nonset", "W6\nBear/\nrecov.", "W7\nBull", "W8\nBull"]
methods = [("Equal-Weight", "#3498db"), ("Markowitz MVO", "#935116"),
           ("S&P 500 B&H", "#17a2b8"), ("Baseline (no gate)", "#d4ac0d"),
           ("Hard Routing", HARD_C), ("Soft MoE (ours)", SOFT_C)]

fig, ax = plt.subplots(figsize=(15, 5.8))
n_m = len(methods)
group_w, group_spacing = 0.84, 1.5
bar_w = group_w / n_m
x = np.arange(8) * group_spacing
for i, (name, color) in enumerate(methods):
    vals = [by[name][w] for w in windows]
    offset = (i - (n_m - 1) / 2) * bar_w
    hl = name in ("Hard Routing", "Soft MoE (ours)", "Baseline (no gate)")
    label = name.replace("Hard Routing", "Hard RA-RL").replace("Soft MoE (ours)", "Soft RA-RL")
    ax.bar(x + offset, vals, width=bar_w * 0.95, color=color, label=label,
           edgecolor="#333" if hl else "#999", linewidth=1.1 if hl else 0.5,
           alpha=1.0 if hl else 0.8, zorder=3 if hl else 2)
ax.axhline(0, color="#444", lw=0.9, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(wlabels, fontsize=9)
ax.set_ylabel("Annualised Sharpe Ratio")
ax.set_title("Per-Window Sharpe, Corrected Mechanics (8-Window Walk-Forward, 2 seeds)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for xi in x[:-1]:
    ax.axvline(xi + group_spacing / 2, color="#ddd", lw=0.8, zorder=0)
fig.tight_layout()
fig.savefig("walkforward_sharpe_corrected.png", dpi=300, bbox_inches="tight")
fig.savefig("walkforward_sharpe_corrected.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved seed_sharpe_dots, bar_corrected_a2c, equity_curve_corrected_seed1, walkforward_sharpe_corrected")
