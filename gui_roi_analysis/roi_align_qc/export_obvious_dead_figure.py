"""Save a side-by-side figure of the obvious-dead rule on real first-pre images.

Columns are the raw max projection, the bleb map, the image after those blebs
are removed, and the dendrite-shaft skeleton. Lengths on the figure are
micrometers.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from obvious_dead import BLEB_MIN, SHAFT_MAX_UM, analyze  # noqa: E402
from qc_assignment_source import SESSION_PKLS  # noqa: E402
from score_obvious_dead import load_first_pre  # noqa: E402
from score_qc_criteria import FEATURE_CSV  # noqa: E402

OUT_PATH = (
    r"//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/obvious_dead_figure.png"
)
SCALE_BAR_UM = 5.0
EXAMPLES = (
    ("20260909_AP5_13_pos1__highmag_7_set0.png", "Obvious death"),
    ("20260909_AP5_13_pos1__highmag_6_set1.png", "Shaft still present"),
    ("20260623_3_pos1__highmag_5_set2.png", "False call (labeled keep)"),
)


def _record_for(features: pd.DataFrame, name: str):
    matched = features[features["image_rel"].astype(str).map(os.path.basename) == name]
    if matched.empty:
        raise KeyError(name)
    return next(matched.itertuples(index=False))


def _scale_bar(ax, xy_um: float, height: int) -> None:
    length_px = SCALE_BAR_UM / xy_um
    y_pos = height - 8
    x_pos = 8
    ax.plot([x_pos, x_pos + length_px], [y_pos, y_pos], color="white", lw=2, solid_capstyle="butt")
    ax.text(x_pos, y_pos - 6, f"{SCALE_BAR_UM:.0f} um", color="white", fontsize=8, va="bottom")


def _show(ax, image: np.ndarray, title: str, cmap: str = "gray") -> None:
    ax.imshow(image, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def render(feature_csv: str = FEATURE_CSV, out_path: str = OUT_PATH) -> str:
    """Write the comparison figure and return its path."""
    features = pd.read_csv(feature_csv)
    pooled = {session: pd.read_pickle(path) for session, path in SESSION_PKLS.items() if os.path.isfile(path)}
    figure, axes = plt.subplots(len(EXAMPLES), 4, figsize=(12.5, 9.2), dpi=140)
    for row, (name, label) in enumerate(EXAMPLES):
        image, xy_um = load_first_pre(_record_for(features, name), pooled)
        result = analyze(image, xy_um)
        rejected = result.bleb >= BLEB_MIN and result.shaft_um <= SHAFT_MAX_UM
        decision = "REJECT" if rejected else "KEEP"
        axes[row, 0].set_ylabel(
            f"{label}\n{decision}\nbleb {result.bleb:.2f}\nshaft {result.shaft_um:.1f} um",
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
            labelpad=36,
            color="#b00020" if rejected else "#1b7a2a",
        )
        _show(axes[row, 0], result.scaled, "1. First pre" if row == 0 else "")
        _scale_bar(axes[row, 0], result.xy_um, result.scaled.shape[0])
        _show(axes[row, 1], result.bleb_response, "2. Bleb map" if row == 0 else "", cmap="magma")
        for y_coord, x_coord, radius_um in result.bleb_centers:
            axes[row, 1].add_patch(
                Circle(
                    (x_coord, y_coord),
                    radius_um / result.xy_um,
                    fill=False,
                    edgecolor="cyan",
                    lw=0.8,
                )
            )
        _show(axes[row, 2], result.residual, "3. Bleb removed" if row == 0 else "")
        overlay = np.stack([result.scaled, result.scaled, result.scaled], axis=-1)
        other = result.skeleton & ~result.shaft_mask
        overlay[other] = (0.35, 0.75, 1.0)
        overlay[result.shaft_mask] = (1.0, 0.55, 0.1)
        _show(axes[row, 3], overlay, "4. Longest shaft (orange)" if row == 0 else "")
    figure.suptitle(
        f"Reject when bleb >= {BLEB_MIN:.2f} and shaft <= {SHAFT_MAX_UM:.2f} um"
        "\nCyan circles are removed blebs. Orange is the shaft length that is measured.",
        fontsize=12,
    )
    figure.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    figure.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_path


def main() -> None:
    print(render())


if __name__ == "__main__":
    main()
