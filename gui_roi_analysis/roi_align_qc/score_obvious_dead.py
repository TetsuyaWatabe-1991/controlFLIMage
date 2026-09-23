"""Score first-pre max projections for an obvious bleb without a shaft."""

from __future__ import annotations

import os
import sys

import pandas as pd
import tifffile

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from export_unhealthy_pre1 import first_pre  # noqa: E402
from obvious_dead import (  # noqa: E402
    BLEB_MIN,
    SHAFT_MAX_UM,
    dead_scores,
    obviously_dead,
    xy_um_from_state,
)
from qc_assignment_source import SESSION_PKLS, normalize_set_label  # noqa: E402
from score_qc_criteria import FEATURE_CSV  # noqa: E402

OBVIOUS = {
    "20260909_AP5_13_pos1__highmag_7_set1.png",
    "20260909_AP5_13_pos1__highmag_2_set0.png",
    "20260909_AP5_13_pos1__highmag_2_set1.png",
    "20260909_AP5_13_pos1__highmag_2_set2.png",
    "20260909_AP5_13_pos1__highmag_2_set3.png",
    "20260909_AP5_13_pos1__highmag_5_set0.png",
    "20260909_AP5_13_pos1__highmag_5_set1.png",
    "20260909_AP5_13_pos1__highmag_5_set2.png",
    "20260909_AP5_13_pos1__highmag_5_set3.png",
    "20260909_AP5_13_pos1__highmag_7_set0.png",
}


def load_first_pre(record, pooled: dict[str, pd.DataFrame]):
    table = pooled[str(record.session)]
    set_label = normalize_set_label(record.set_label)
    matched = table[
        (table["group"].astype(str) == str(record.group))
        & (table["nth_set_label"].map(normalize_set_label) == set_label)
    ]
    after = next(path for path in matched["after_align_full_save_path"] if isinstance(path, str) and path)
    state = matched.loc[matched["after_align_full_save_path"] == after, "statedict"].iloc[0]
    base = os.path.splitext(after)[0]
    info = pd.read_csv(base + "_frame_info.csv")
    stack = tifffile.imread(after)
    mask = tifffile.imread(base + "_Spine_roi_mask.tif")
    image, _roi = first_pre(info, stack, mask)
    return image, xy_um_from_state(state)


def score_rows(feature_csv: str = FEATURE_CSV) -> pd.DataFrame:
    features = pd.read_csv(feature_csv)
    chosen = features[features["category"].isin([8, 9])]
    pooled = {session: pd.read_pickle(path) for session, path in SESSION_PKLS.items() if os.path.isfile(path)}
    rows = []
    for number, record in enumerate(chosen.itertuples(index=False), start=1):
        image, xy_um = load_first_pre(record, pooled)
        bleb, shaft_um = dead_scores(image, xy_um)
        name = os.path.basename(str(record.image_rel))
        if name in OBVIOUS:
            kind = "obvious"
        elif int(record.category) == 8:
            kind = "maybe"
        else:
            kind = "keep"
        rows.append(
            {
                "kind": kind,
                "name": name,
                "xy_um": xy_um,
                "bleb": bleb,
                "shaft_um": shaft_um,
                "called_dead": obviously_dead(image, xy_um),
            }
        )
        if number % 40 == 0 or number == len(chosen):
            print(f"scored {number}/{len(chosen)}", flush=True)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> str:
    lines = [f"cutoff bleb>={BLEB_MIN} and shaft<={SHAFT_MAX_UM:.2f} um", ""]
    for kind in ("obvious", "maybe", "keep"):
        part = frame[frame["kind"] == kind]
        hit = int(part["called_dead"].sum())
        lines.append(f"{kind:8} {hit}/{len(part)} called dead")
    lines.append("")
    lines.append(frame[frame["kind"] != "keep"].sort_values(["kind", "bleb"]).to_string(index=False))
    keeps_all = frame[frame["kind"] == "keep"]
    lines.append("")
    lines.append(
        "keep bleb "
        + " ".join(f"p{int(q * 100)}={keeps_all['bleb'].quantile(q):.2f}" for q in (0.5, 0.9, 0.95, 1.0))
    )
    lines.append(
        "keep shaft_um "
        + " ".join(f"p{int(q * 100)}={keeps_all['shaft_um'].quantile(q):.2f}" for q in (0.5, 0.1, 0.05, 0.0))
    )
    keeps = keeps_all[keeps_all["called_dead"]]
    lines.append(f"keeps called dead: {len(keeps)}")
    if not keeps.empty:
        lines.append(keeps.sort_values("bleb", ascending=False).head(20).to_string(index=False))
    return "\n".join(lines)


def main() -> None:
    frame = score_rows()
    print(summarize(frame))


if __name__ == "__main__":
    main()
