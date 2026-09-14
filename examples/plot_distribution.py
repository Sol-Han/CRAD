"""
Reproduce the CRAD label-distribution figures from the consolidated CSV.

No CROCS/CRACCS dependency: only pandas/numpy/matplotlib. Run:
    python plot_distribution.py
Figures are written next to this script's ../figures/ directory.

Note on sign convention: these labels match the CRAD paper's convention,
which is the opposite sign from the annotation tool's own live slider
display -- see the main README for details.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
LABELS = os.path.join(HERE, "..", "labels", "gt_labels.csv")
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

TRANGE = {
    "pohang00": (580, 2170), "pohang01": (690, 2640), "pohang02": (840, 2240),
    "pohang03": (800, 2420), "pohang04": (650, 2200), "pohang05": (470, 2170),
}
IR_SEQS = {"pohang01", "pohang05"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

df = pd.read_csv(LABELS)
df["t_norm"] = df.apply(
    lambda r: (r["frame_id"] / 10.0 - TRANGE[r["seq"]][0]) / (TRANGE[r["seq"]][1] - TRANGE[r["seq"]][0]),
    axis=1).clip(0, 1)
df["group"] = df["seq"].apply(lambda s: "Night / IR" if s in IR_SEQS else "Day / RGB")
rgb, ir = df[df.group == "Day / RGB"], df[df.group == "Night / IR"]


def style_axes(ax):
    ax.tick_params(colors=SECONDARY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---- Figure 1: roll/pitch joint distribution, Day/RGB vs Night/IR ----
fig = plt.figure(figsize=(3.4, 3.4))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 0.42], height_ratios=[0.42, 4],
                        wspace=0.04, hspace=0.04, left=0.16, right=0.97, top=0.97, bottom=0.13)
ax_main = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

ax_main.scatter(rgb.roll_deg, rgb.pitch_deg, s=13, marker="o", facecolor=BLUE,
                 edgecolor="none", alpha=0.5, label=f"Day / RGB (n={len(rgb)})", zorder=3)
ax_main.scatter(ir.roll_deg, ir.pitch_deg, s=15, marker="^", facecolor=ORANGE,
                 edgecolor="none", alpha=0.55, label=f"Night / IR (n={len(ir)})", zorder=4)
ax_main.axhline(0, color=GRID, lw=0.6, zorder=1)
ax_main.axvline(0, color=GRID, lw=0.6, zorder=1)
ax_main.set_xlabel("Roll [deg]", color=INK)
ax_main.set_ylabel("Pitch [deg]", color=INK)
style_axes(ax_main)
leg = ax_main.legend(loc="upper right", fontsize=6.3, frameon=False, handletextpad=0.4,
                      borderaxespad=0.2, markerscale=1.3)
for t in leg.get_texts():
    t.set_color(SECONDARY)

bins_roll = np.linspace(df.roll_deg.min() - 0.2, df.roll_deg.max() + 0.2, 26)
bins_pitch = np.linspace(df.pitch_deg.min() - 0.2, df.pitch_deg.max() + 0.2, 26)
ax_top.hist(rgb.roll_deg, bins=bins_roll, color=BLUE, alpha=0.55, lw=0)
ax_top.hist(ir.roll_deg, bins=bins_roll, color=ORANGE, alpha=0.55, lw=0)
ax_right.hist(rgb.pitch_deg, bins=bins_pitch, color=BLUE, alpha=0.55, lw=0, orientation="horizontal")
ax_right.hist(ir.pitch_deg, bins=bins_pitch, color=ORANGE, alpha=0.55, lw=0, orientation="horizontal")
top_max = max(np.histogram(rgb.roll_deg, bins=bins_roll)[0].max(),
              np.histogram(ir.roll_deg, bins=bins_roll)[0].max())
right_max = max(np.histogram(rgb.pitch_deg, bins=bins_pitch)[0].max(),
                np.histogram(ir.pitch_deg, bins=bins_pitch)[0].max())
ax_top.set_ylim(0, top_max * 1.08)
ax_right.set_xlim(0, right_max * 1.08)
for ax in (ax_top, ax_right):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False,
                    labeltop=False, labelright=False)
ax_top.set_yticks([]); ax_right.set_xticks([])
ax_top.margins(x=0); ax_right.margins(y=0)

fig.savefig(os.path.join(FIGDIR, "label_distribution.png"), dpi=200)
fig.savefig(os.path.join(FIGDIR, "label_distribution.pdf"))
plt.close(fig)

# ---- Figure 2: pitch vs. normalized sequence progress ----
fig2, ax2 = plt.subplots(figsize=(3.4, 2.7))
fig2.subplots_adjust(left=0.135, right=0.965, top=0.95, bottom=0.155)
ax2.scatter(rgb.t_norm, rgb.pitch_deg, s=10, marker="o", facecolor=BLUE, edgecolor="none", alpha=0.55, zorder=2)
ax2.scatter(ir.t_norm, ir.pitch_deg, s=12, marker="^", facecolor=ORANGE, edgecolor="none", alpha=0.6, zorder=2)
ax2.set_xlabel("Normalized sequence progress (0 = start, 1 = end)", color=INK, fontsize=7.3)
ax2.set_ylabel("Pitch [deg]", color=INK)
xpad = 0.03
ypad = (df.pitch_deg.max() - df.pitch_deg.min()) * 0.06
ax2.set_xlim(-xpad, 1 + xpad)
ax2.set_ylim(df.pitch_deg.min() - ypad, df.pitch_deg.max() + ypad)
style_axes(ax2)
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="none", markersize=5, label="Day / RGB"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor=ORANGE, markeredgecolor="none", markersize=5.5, label="Night / IR"),
]
leg2 = ax2.legend(handles=handles, loc="lower right" if df.pitch_deg.corr(df.t_norm) > 0 else "upper right",
                   fontsize=6.3, frameon=False, handletextpad=0.4, borderaxespad=0.2)
for t in leg2.get_texts():
    t.set_color(SECONDARY)

fig2.savefig(os.path.join(FIGDIR, "pitch_time_progress.png"), dpi=200)
fig2.savefig(os.path.join(FIGDIR, "pitch_time_progress.pdf"))
plt.close(fig2)

r = df.pitch_deg.corr(df.t_norm)
print(f"pitch vs. normalized progress: r = {r:.3f}")
print("wrote:", os.path.join(FIGDIR, "label_distribution.png"))
print("wrote:", os.path.join(FIGDIR, "pitch_time_progress.png"))
