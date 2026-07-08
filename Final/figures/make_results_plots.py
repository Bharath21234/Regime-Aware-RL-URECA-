"""Generates Figures 6, 7, 8 for Section 5 (Results):
  Fig 6 - Table I bar chart (pooled n=9, Hard RA-RL vs Soft RA-RL)
  Fig 7 - Table II bar chart (n=3, Hard/Soft RA-RL/Learned Router)
  Fig 8 - Walk-forward per-window Sharpe line chart (Table III / A3)
All values are taken directly from the tables already in the paper (and the
recomputed Table II stds verified against the underlying per-seed data).
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"

HARD_C = "#c0392b"
SOFT_C = "#1d8a4e"
ROUTER_C = "#7f7f7f"

# ----------------------------------------------------------------------------
# Figure 6 - Table I (pooled n=9)
# ----------------------------------------------------------------------------
metrics = ["Return (%)", "Sharpe", "Max Drawdown (%)", "Sortino"]
hard_mean = [12.36, 0.349, 36.17, 0.350]
hard_std = [14.60, 0.258, 10.13, 0.259]
soft_mean = [28.07, 0.591, 23.80, 0.618]
soft_std = [19.66, 0.221, 5.74, 0.232]

fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
for ax, m, hm, hs, sm, ss in zip(axes, metrics, hard_mean, hard_std, soft_mean, soft_std):
    x = [0, 1]
    vals = [hm, sm]
    errs = [hs, ss]
    colors = [HARD_C, SOFT_C]
    ax.bar(x, vals, yerr=errs, capsize=5, color=colors, edgecolor="#333", width=0.6,
           error_kw={"elinewidth": 1.3})
    ax.set_xticks(x)
    ax.set_xticklabels(["Hard\nRA-RL", "Soft\nRA-RL"], fontsize=10)
    ax.set_title(m, fontsize=11, fontweight="bold")
    ax.axhline(0, color="#999", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Table I — Pooled Comparison, Hard RA-RL vs Soft RA-RL (n=9 seeds, mean ± std)",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig("bar_table1.png", dpi=300, bbox_inches="tight")
fig.savefig("bar_table1.pdf", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 7 - Table II (n=3, Run B)
# ----------------------------------------------------------------------------
metrics2 = ["Return (%)", "Sharpe", "Max Drawdown (%)", "Sortino"]
hard2_mean = [4.89, 0.222, 43.23, 0.215]
hard2_std = [19.75, 0.334, 11.18, 0.323]
soft2_mean = [22.97, 0.568, 20.70, 0.609]
soft2_std = [5.39, 0.121, 3.64, 0.118]
router_mean = [-5.07, -0.060, 34.15, -0.049]
router_std = [20.76, 0.456, 5.91, 0.452]

fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))
for ax, m, hm, hs, sm, ss, rm, rs in zip(axes, metrics2, hard2_mean, hard2_std,
                                          soft2_mean, soft2_std, router_mean, router_std):
    x = [0, 1, 2]
    vals = [hm, sm, rm]
    errs = [hs, ss, rs]
    colors = [HARD_C, SOFT_C, ROUTER_C]
    ax.bar(x, vals, yerr=errs, capsize=5, color=colors, edgecolor="#333", width=0.6,
           error_kw={"elinewidth": 1.3})
    ax.set_xticks(x)
    ax.set_xticklabels(["Hard\nRA-RL", "Soft\nRA-RL", "Learned\nRouter"], fontsize=9.5)
    ax.set_title(m, fontsize=11, fontweight="bold")
    ax.axhline(0, color="#999", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Table II — Learned Router Ablation (n=3 seeds, mean ± std)",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig("bar_table2.png", dpi=300, bbox_inches="tight")
fig.savefig("bar_table2.pdf", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 8 - Walk-forward per-window Sharpe (Table III / A3)
# ----------------------------------------------------------------------------
windows = ["W1\nCOVID", "W2\nRecov.", "W3\nBull", "W4\nBull", "W5\nBear\nonset",
           "W6\nBear/\nrecov.", "W7\nBull", "W8\nBull"]
data = {
    "Equal-Weight":   ([0.33, 3.18, 3.10, 2.03, -1.47, 0.92, 2.87, 2.21], "#3498db"),
    "Markowitz MVO":  ([1.15, 0.95, 0.33, 3.26, -1.53, -0.06, 3.39, 1.56], "#935116"),
    "S&P 500 B&H":    ([0.03, 2.43, 2.45, 1.72, -1.72, 0.22, 2.33, 1.38], "#17a2b8"),
    "Baseline":       ([-0.44, 1.97, -0.16, 1.40, -1.90, 0.50, -0.95, 2.36], "#d4ac0d"),
    "LSTM-Context":   ([0.96, 1.91, 2.35, 2.21, -2.91, -0.05, 3.80, 1.89], "#8e44ad"),
    "Hard RA-RL":     ([0.05, 1.76, 1.85, 1.76, -1.04, 2.83, 0.54, 1.67], HARD_C),
    "Soft RA-RL":     ([1.00, 1.83, 2.45, 2.23, -1.77, 0.20, 3.07, 1.80], SOFT_C),
}

fig, ax = plt.subplots(figsize=(16, 6.2))
n_methods = len(data)
group_w = 0.84
bar_w = group_w / n_methods
group_spacing = 1.6
x = np.arange(8) * group_spacing

for i, (name, (vals, color)) in enumerate(data.items()):
    offset = (i - (n_methods - 1) / 2) * bar_w
    highlight = name in ("Hard RA-RL", "Soft RA-RL")
    ax.bar(x + offset, vals, width=bar_w * 0.95, color=color, label=name,
           edgecolor="#333" if highlight else "#999",
           linewidth=1.2 if highlight else 0.5,
           alpha=1.0 if highlight else 0.8,
           zorder=3 if highlight else 2)

ax.axhline(0, color="#444", lw=0.9, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(windows, fontsize=9)
ax.set_ylabel("Annualised Sharpe Ratio")
ax.set_title("Per-Window Sharpe Ratio, All Methods (8-Window Walk-Forward Validation)",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for xi in x[:-1]:
    ax.axvline(xi + group_spacing / 2, color="#ddd", lw=0.8, zorder=0)
ax.set_xlim(x[0] - group_spacing / 2, x[-1] + group_spacing / 2)

fig.tight_layout()
fig.savefig("walkforward_sharpe.png", dpi=300, bbox_inches="tight")
fig.savefig("walkforward_sharpe.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved bar_table1, bar_table2, walkforward_sharpe ({png,pdf})")
