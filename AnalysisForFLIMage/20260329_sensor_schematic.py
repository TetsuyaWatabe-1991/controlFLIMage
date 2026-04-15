# -*- coding: utf-8 -*-
"""
Sensor construct schematic diagram generator.
Produces a vector PDF suitable for publication figures.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

matplotlib.rc('pdf', fonttype=42)
plt.rcParams["font.family"] = "Arial"

# ── Construct definitions ──────────────────────────────────────────────────────
# Each domain: label shown inside box, width (relative), fill color, optional
# arrow_label drawn above the box with a curved arrow.
CONSTRUCTS = [
    {
        "row_label": "p38 sensor",
        "domains": [
            {"label": "sREACh",   "width": 0.7, "color": "#8B8000", "fontsize": 6.5},
            {"label": "WW\ndomain","width": 0.55,"color": "#D98080", "fontsize": 6,
             "arrow_label": "WW domain"},
            {"label": "EV Linker","width": 3.4, "color": "#9E9E9E", "fontsize": 7},
            {"label": "",         "width": 0.85,"color": "#7B4FA6", "fontsize": 7,
             "arrow_label": "p38 substrate & docking site"},
            {"label": "EGFP",    "width": 0.7, "color": "#4CAF50", "fontsize": 7},
            {"label": "",        "width": 0.45,"color": "#E07000", "fontsize": 7,
             "arrow_label": "NES"},
        ],
    },
    {
        "row_label": "p38 Ser to Thr",
        "row_sublabel": "sREACh\nS208F/R223F/V224L",
        "domains": [
            {"label": "sREACh",   "width": 0.7, "color": "#8B8000", "fontsize": 6.5},
            {"label": "WW\ndomain","width": 0.55,"color": "#D98080", "fontsize": 6,
             "arrow_label": "WW domain"},
            {"label": "EV Linker","width": 3.4, "color": "#9E9E9E", "fontsize": 7},
            {"label": "",         "width": 0.85,"color": "#7B4FA6", "fontsize": 7,
             "arrow_label": "p38 substrate & docking site"},
            {"label": "EGFP",    "width": 0.7, "color": "#4CAF50", "fontsize": 7},
            {"label": "",        "width": 0.45,"color": "#E07000", "fontsize": 7,
             "arrow_label": "NES"},
        ],
    },
]

# ── Layout constants ───────────────────────────────────────────────────────────
ROW_HEIGHT      = 0.30   # height of each domain box
ROW_GAP         = 1.10   # vertical distance between row centres (data units)
ARROW_Y_OFFSET  = 0.40   # how far above box top the arrow tip sits
LABEL_Y_OFFSET  = 0.60   # how far above box top the text sits
FIG_W, FIG_H    = 7.0, 2.8

# ── Draw ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_aspect('equal')
ax.axis('off')

total_width = sum(d["width"] for d in CONSTRUCTS[0]["domains"])
n_rows      = len(CONSTRUCTS)
total_height= (n_rows - 1) * ROW_GAP + ROW_HEIGHT + 1.5

ax.set_xlim(-0.3, total_width + 2.5)
ax.set_ylim(-0.6, total_height)

for row_idx, construct in enumerate(CONSTRUCTS):
    y_centre = (n_rows - 1 - row_idx) * ROW_GAP + ROW_HEIGHT / 2 + 0.6

    x = 0.0
    # Track which domains already have their arrow label drawn
    drawn_arrow_labels: set[str] = set()

    for dom in construct["domains"]:
        w = dom["width"]

        # Domain rectangle
        rect = mpatches.FancyBboxPatch(
            (x, y_centre - ROW_HEIGHT / 2),
            w, ROW_HEIGHT,
            boxstyle="square,pad=0",
            linewidth=0.6, edgecolor="white",
            facecolor=dom["color"],
        )
        ax.add_patch(rect)

        # Label inside box
        if dom["label"]:
            ax.text(
                x + w / 2, y_centre,
                dom["label"],
                ha="center", va="center",
                fontsize=dom.get("fontsize", 7),
                color="black", fontweight="normal",
            )

        # Arrow + label above box (avoid duplicates)
        arrow_lbl = dom.get("arrow_label", "")
        if arrow_lbl and arrow_lbl not in drawn_arrow_labels:
            drawn_arrow_labels.add(arrow_lbl)
            cx = x + w / 2   # x centre of this domain

            # Vertical arrow
            ax.annotate(
                "",
                xy=(cx, y_centre + ROW_HEIGHT / 2),
                xytext=(cx, y_centre + ROW_HEIGHT / 2 + ARROW_Y_OFFSET),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="black",
                    lw=0.8,
                    mutation_scale=7,
                ),
            )

            # Text above arrow
            ax.text(
                cx,
                y_centre + ROW_HEIGHT / 2 + LABEL_Y_OFFSET,
                arrow_lbl,
                ha="center", va="bottom",
                fontsize=7.5,
                color="black",
            )

        x += w

    # Construct name to the right of the row
    row_label    = construct.get("row_label", "")
    row_sublabel = construct.get("row_sublabel", "")
    ax.text(
        x + 0.15, y_centre,
        row_label,
        ha="left", va="center" if not row_sublabel else "top",
        fontsize=8, fontstyle="italic",
    )
    if row_sublabel:
        ax.text(
            x + 0.15, y_centre - 0.18,
            row_sublabel,
            ha="left", va="top",
            fontsize=6.5, color="#444444",
        )

plt.tight_layout(pad=0.3)

save_path = r"C:\Users\WatabeT\Documents\Git\controlFLIMage\AnalysisForFLIMage\20260329_sensor_schematic.pdf"
plt.savefig(save_path, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()
