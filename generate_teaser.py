"""Generate DPS+ method overview teaser figure."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Color palette
C_BG = "#FAFAFA"
C_DARK = "#1E1B4B"
C_PRIMARY = "#6366F1"
C_LIGHT = "#A78BFA"
C_ACCENT = "#F59E0B"
C_GREEN = "#10B981"
C_WHITE = "#FFFFFF"
C_GRAY = "#6B7280"
C_LIGHT_GRAY = "#F3F4F6"
C_DANGER = "#EF4444"
C_BLUE_LIGHT = "#DBEAFE"

fig = plt.figure(figsize=(18, 10), facecolor=C_WHITE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")


def draw_box(ax, x, y, w, h, color=C_WHITE, edge=C_PRIMARY, lw=2, radius=0.15,
             text="", text_color=C_DARK, fontsize=10, fontweight="normal", ha="center", va="center"):
    """Draw a rounded rectangle with text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius*min(w,h)}",
                         facecolor=color, edgecolor=edge, linewidth=lw, zorder=3)
    ax.add_patch(box)
    if text:
        ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, fontweight=fontweight,
                color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.5, style="simple", zorder=1):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle="arc3,rad=0"), zorder=zorder)


def draw_curved_arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.5, rad=0.3, zorder=1):
    """Draw a curved arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=zorder)


# ═══════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════
ax.text(9, 9.5, "DPS+: Defensive Perturbation Suppression via Weak-to-Strong Collaboration",
        ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)
ax.text(9, 9.0, "Method Overview",
        ha="center", va="center", fontsize=12, fontweight="normal", color=C_GRAY)

# ═══════════════════════════════════════════════════════════
# ROW 1: Image Pipeline (top)
# ═══════════════════════════════════════════════════════════
row1_y = 7.3
box_w, box_h = 2.0, 1.6
gap = 1.0

# Stage labels
label_y = row1_y + box_h/2 + 0.35

# Image 1: Adversarial Input
x1 = 2.5
draw_box(ax, x1, row1_y, box_w, box_h, color="#FEE2E2", edge=C_DANGER, lw=2,
         text="Adversarial\nImage", fontsize=10, fontweight="bold", text_color=C_DANGER)
ax.text(x1, label_y, "Input", ha="center", fontsize=9, color=C_GRAY, fontweight="bold")

# Arrow
draw_arrow(ax, x1 + box_w/2 + 0.15, row1_y, x1 + box_w/2 + gap - 0.15, row1_y, color=C_GRAY)

# Image 2: Heatmap
x2 = x1 + box_w + gap
draw_box(ax, x2, row1_y, box_w, box_h, color="#EDE9FE", edge=C_PRIMARY, lw=2,
         text="Attention\nHeatmap", fontsize=10, fontweight="bold", text_color=C_PRIMARY)
ax.text(x2, label_y, "Local Model", ha="center", fontsize=9, color=C_GRAY, fontweight="bold")

# Arrow
draw_arrow(ax, x2 + box_w/2 + 0.15, row1_y, x2 + box_w/2 + gap - 0.15, row1_y, color=C_GRAY)

# Image 3: Masked image
x3 = x2 + box_w + gap
draw_box(ax, x3, row1_y, box_w, box_h, color="#E0E7FF", edge=C_PRIMARY, lw=2,
         text="Masked\nImage", fontsize=10, fontweight="bold", text_color=C_PRIMARY)
ax.text(x3, label_y, "Masking (Top-k%)", ha="center", fontsize=9, color=C_GRAY, fontweight="bold")

# Small model label between x2 and x3
ax.text((x2 + x3) / 2, row1_y - box_h/2 - 0.35, "Qwen2.5-VL-3B-Instruct",
        ha="center", fontsize=8, color=C_LIGHT, style="italic")

# ═══════════════════════════════════════════════════════════
# ROW 2: Dual-path processing
# ═══════════════════════════════════════════════════════════
row2_y = 5.2

# Left: Strong Model (full image)
left_x = 4.5
draw_box(ax, left_x, row2_y, 4.0, 2.0, color=C_LIGHT_GRAY, edge=C_PRIMARY, lw=2.5, radius=0.08,
         text="", fontsize=11)

ax.text(left_x, row2_y + 0.65, "Strong Model (Remote VLM)", ha="center", fontsize=11,
        fontweight="bold", color=C_PRIMARY)
ax.text(left_x, row2_y + 0.15, "Full-image reasoning\nInitial answer on original image",
        ha="center", fontsize=9, color=C_DARK)

# Right: Weak agent (masked image)
right_x = 13.5
draw_box(ax, right_x, row2_y, 4.0, 2.0, color="#FEF3C7", edge=C_ACCENT, lw=2.5, radius=0.08,
         text="", fontsize=11)

ax.text(right_x, row2_y + 0.65, "Weak Observation Agent", ha="center", fontsize=11,
        fontweight="bold", color=C_ACCENT)
ax.text(right_x, row2_y + 0.15, "Masked-image observation\nDescribe visible content only",
        ha="center", fontsize=9, color=C_DARK)

# Arrows from row1 to row2
draw_arrow(ax, x1, row1_y - box_h/2, left_x - 1.2, row2_y + 1.0, color=C_GRAY, lw=1.2)
draw_arrow(ax, x3, row1_y - box_h/2, right_x + 1.2, row2_y + 1.0, color=C_GRAY, lw=1.2)

# Labels on arrows
ax.text(x1, (row1_y - box_h/2 + row2_y + 1.0) / 2, "full image",
        ha="center", fontsize=7, color=C_GRAY, rotation=90, va="center")
ax.text(x3, (row1_y - box_h/2 + row2_y + 1.0) / 2, "masked only",
        ha="center", fontsize=7, color=C_GRAY, rotation=90, va="center")

# ═══════════════════════════════════════════════════════════
# ROW 3: Cross-validation
# ═══════════════════════════════════════════════════════════
row3_y = 2.5

# Center: Cross-validation box
center_x = 9.0
draw_box(ax, center_x, row3_y, 5.0, 1.6, color="#D1FAE5", edge=C_GREEN, lw=2.5, radius=0.08,
         text="", fontsize=10)

ax.text(center_x, row3_y + 0.45, "Cross-Validation & Re-evaluation", ha="center", fontsize=11,
        fontweight="bold", color="#065F46")
ax.text(center_x, row3_y - 0.15, "Compare initial answer with weak observation\nCorrect adversarial deception if contradiction found",
        ha="center", fontsize=9, color=C_DARK)

# Arrows from row2 to row3
draw_arrow(ax, left_x, row2_y - 1.0, center_x - 1.8, row3_y + 0.8, color=C_PRIMARY, lw=1.5)
draw_arrow(ax, right_x, row2_y - 1.0, center_x + 1.8, row3_y + 0.8, color=C_ACCENT, lw=1.5)

# ═══════════════════════════════════════════════════════════
# ROW 4: Final output
# ═══════════════════════════════════════════════════════════
row4_y = 0.8

draw_box(ax, center_x, row4_y, 4.5, 1.0, color=C_GREEN, edge="#047857", lw=2.5, radius=0.15,
         text="Safe & Accurate Output", fontsize=12, fontweight="bold", text_color=C_WHITE)
# Override text color
ax.texts[-1].set_color("white")

# Arrow from row3 to row4
draw_arrow(ax, center_x, row3_y - 0.8, center_x, row4_y + 0.5, color=C_GREEN, lw=1.8)

# ═══════════════════════════════════════════════════════════
# LEGEND (bottom right)
# ═══════════════════════════════════════════════════════════
legend_items = [
    (C_PRIMARY, "Strong Model Path"),
    (C_ACCENT, "Weak Agent Path"),
    (C_GREEN, "Safe Output"),
    (C_DANGER, "Adversarial Input"),
]
lx, ly = 15.5, 3.6
for i, (color, label) in enumerate(legend_items):
    yi = ly - i * 0.4
    ax.add_patch(plt.Rectangle((lx - 0.25, yi - 0.1), 0.5, 0.2, color=color, zorder=5))
    ax.text(lx + 0.5, yi, label, fontsize=8, color=C_DARK, va="center")

# ═══════════════════════════════════════════════════════════
# STEP NUMBERS (small circles)
# ═══════════════════════════════════════════════════════════
steps = [
    (x1 - box_w/2 - 0.25, row1_y, "1"),
    (x2 - box_w/2 - 0.25, row1_y, "2"),
    (x3 - box_w/2 - 0.25, row1_y, "3"),
    (left_x - 2.25, row2_y, "4"),
    (right_x + 2.25, row2_y, "5"),
    (center_x - 2.75, row3_y, "6"),
    (center_x - 2.55, row4_y, "7"),
]
for sx, sy, num in steps:
    circ = plt.Circle((sx, sy), 0.22, color=C_PRIMARY, zorder=10, clip_on=False)
    ax.add_patch(circ)
    ax.text(sx, sy, num, ha="center", va="center", fontsize=8, fontweight="bold",
            color=C_WHITE, zorder=11)

# Save
output_path = "/Users/lisheng/Desktop/Program/DPS_explanation/DPS+/assets/method_overview.png"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=C_WHITE, edgecolor="none")
plt.close(fig)
print(f"Saved to {output_path}")
