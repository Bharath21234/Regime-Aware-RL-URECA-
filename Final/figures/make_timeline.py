"""Generates Fig. 5: a single timeline showing the single-period train/test
split (Section 4.2) and the 8 walk-forward windows (Section 4.3, Table A1).
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"

TRAIN_COLOR = "#9fb8d9"
TEST_COLORS = {
    "COVID-19 crash": "#e74c3c",
    "Recovery": "#f39c12",
    "Bull": "#2ecc71",
    "Bear onset": "#c0392b",
    "Bear/recovery": "#e67e22",
    "Single-period test": "#8e44ad",
}

# Single-period protocol
single = {
    "train": ("2015-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2024-01-01"),
    "test_label": "Single-period test",
}

# Walk-forward windows (Table A1)
windows = [
    ("W1", "2015-01-01", "2019-12-31", "2020-01-01", "2020-06-30", "COVID-19 crash"),
    ("W2", "2015-01-01", "2020-06-30", "2020-07-01", "2020-12-31", "Recovery"),
    ("W3", "2015-01-01", "2020-12-31", "2021-01-01", "2021-06-30", "Bull"),
    ("W4", "2015-01-01", "2021-06-30", "2021-07-01", "2021-12-31", "Bull"),
    ("W5", "2015-01-01", "2021-12-31", "2022-01-01", "2022-06-30", "Bear onset"),
    ("W6", "2015-01-01", "2022-06-30", "2022-07-01", "2022-12-31", "Bear/recovery"),
    ("W7", "2015-01-01", "2022-12-31", "2023-01-01", "2023-06-30", "Bull"),
    ("W8", "2015-01-01", "2023-06-30", "2023-07-01", "2023-12-31", "Bull"),
]

fig, ax = plt.subplots(figsize=(12, 6.5))

rows = []
row_labels = []

# Single-period protocol row (drawn first, at top)
rows.append(("Single-Period", single["train"], single["test"], single["test_label"]))
row_labels.append("Single-Period\nProtocol")

for wid, tr_s, tr_e, te_s, te_e, ctx in windows:
    rows.append((wid, (tr_s, tr_e), (te_s, te_e), ctx))
    row_labels.append(wid)

n = len(rows)
bar_h = 0.6

for i, (name, train, test, ctx) in enumerate(rows):
    y = n - i  # top to bottom
    t0, t1 = pd.Timestamp(train[0]), pd.Timestamp(train[1])
    s0, s1 = pd.Timestamp(test[0]), pd.Timestamp(test[1])
    ax.barh(y, (t1 - t0).days, left=t0, height=bar_h, color=TRAIN_COLOR,
            edgecolor="#34495e", linewidth=0.8, zorder=2)
    color = TEST_COLORS.get(ctx, "#888")
    ax.barh(y, (s1 - s0).days, left=s0, height=bar_h, color=color,
            edgecolor="#34495e", linewidth=0.8, zorder=2)

ax.set_yticks([n - i for i in range(n)])
ax.set_yticklabels(row_labels, fontsize=9.5)
ax.axhline(n - 0.5, color="#999", lw=0.8, ls="--")  # separator after single-period row

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(pd.Timestamp("2014-09-01"), pd.Timestamp("2024-04-01"))
ax.set_ylim(0.3, n + 0.8)

handles = [
    Patch(facecolor=TRAIN_COLOR, edgecolor="#34495e", label="Train period"),
    Patch(facecolor=TEST_COLORS["COVID-19 crash"], edgecolor="#34495e", label="Test: COVID-19 crash"),
    Patch(facecolor=TEST_COLORS["Recovery"], edgecolor="#34495e", label="Test: Recovery / Bear-onset / Bear-recovery"),
    Patch(facecolor=TEST_COLORS["Bull"], edgecolor="#34495e", label="Test: Bull"),
    Patch(facecolor=TEST_COLORS["Single-period test"], edgecolor="#34495e", label="Test: Single-period (Bear 2022 -> Bull 2023)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
          ncol=2, fontsize=9, frameon=False)

ax.set_title("Train/Test Timeline: Single-Period Protocol and 8-Window Walk-Forward Validation",
             fontsize=12.5, fontweight="bold", pad=12)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.tight_layout()
fig.savefig("timeline.png", dpi=300, bbox_inches="tight")
fig.savefig("timeline.pdf", bbox_inches="tight")
print("Saved timeline.{png,pdf}")
