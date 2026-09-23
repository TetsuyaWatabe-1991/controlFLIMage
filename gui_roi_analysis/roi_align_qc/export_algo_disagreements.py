"""Save sheets whose reject/include call changed, with the rule that fired.

The algorithm rejects when any of three cutoffs is crossed. Each cutoff was
chosen so that rule alone keeps at least 90 percent of the human keeps.
Positive class is Reject.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from qc_label_store import CATEGORIES  # noqa: E402

SHEET_ROOT = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/reject_qc"
OUT_DIR = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/algo_reclass"
FEATURE_CSV = os.path.join(_THIS, "qc_criteria_features.csv")

SHIFT_CUTOFF = 3.132491021535417
CORR_CUTOFF = 0.4234618356340035
UNC_CUTOFF = 2.738600015640259

FOLDERS = (
    "prior_reject_now_include",
    "prior_include_now_reject",
    "false_positive",
    "false_negative",
)
TITLES = {
    "prior_reject_now_include": "PRIOR REJECT -> INCLUDE",
    "prior_include_now_reject": "PRIOR INCLUDE -> REJECT",
    "false_positive": "FALSE POSITIVE    human keep, algorithm reject",
    "false_negative": "FALSE NEGATIVE    human reject, algorithm include",
}


def rule_lines(shift: float, corr: float, unc_dist: float) -> list[tuple[bool, str]]:
    """Three measurements. The first element is whether that rule rejects."""
    return [
        (
            bool(np.isfinite(shift) and shift >= SHIFT_CUTOFF),
            f"assignment shift {_fmt(shift, 2)} px    cutoff {SHIFT_CUTOFF:.2f} px",
        ),
        (
            bool(np.isfinite(corr) and corr <= CORR_CUTOFF),
            f"assignment correlation {_fmt(corr, 3)}    cutoff {CORR_CUTOFF:.3f}",
        ),
        (
            bool(np.isfinite(unc_dist) and unc_dist >= UNC_CUTOFF),
            f"uncaging distance {_fmt(unc_dist, 2)} px    cutoff {UNC_CUTOFF:.2f} px",
        ),
    ]


def algorithm_rejects(shift: float, corr: float, unc_dist: float) -> bool:
    return any(fired for fired, _text in rule_lines(shift, corr, unc_dist))


def disagreement_folders(prior_pool: str, human_category: int, algo_reject: bool) -> list[str]:
    """Folders for a changed prior call, and for errors against the human label."""
    found = []
    if prior_pool == "reject" and not algo_reject:
        found.append("prior_reject_now_include")
    if prior_pool == "keep" and algo_reject:
        found.append("prior_include_now_reject")
    if human_category == 9 and algo_reject:
        found.append("false_positive")
    if human_category != 9 and not algo_reject:
        found.append("false_negative")
    return found


def annotate_sheet(image: np.ndarray, title: str, human_line: str, rules: list[tuple[bool, str]]) -> np.ndarray:
    """Stack a reason banner above the QC sheet."""
    width = image.shape[1]
    line_h = 26
    banner_h = 12 + line_h * (2 + len(rules))
    banner = np.full((banner_h, width, 3), 32, np.uint8)
    cv2.putText(banner, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, human_line, (10, 24 + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    for index, (fired, text) in enumerate(rules):
        prefix = "FIRED  " if fired else "ok     "
        color = (0, 180, 255) if fired else (170, 170, 170)
        y_coord = 24 + line_h * (2 + index)
        cv2.putText(banner, prefix + text, (10, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([banner, image])


def _fmt(value: float, digits: int) -> str:
    if not np.isfinite(value):
        return "missing"
    return f"{float(value):.{digits}f}"


def export_disagreements(feature_csv: str = FEATURE_CSV, sheet_root: str = SHEET_ROOT, out_dir: str = OUT_DIR) -> pd.DataFrame:
    features = pd.read_csv(feature_csv)
    rows = []
    for folder in FOLDERS:
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
    for record in features.itertuples(index=False):
        rules = rule_lines(record.assign_shift_px, record.assign_corr, record.unc_dist_px)
        algo_reject = any(fired for fired, _text in rules)
        prior = str(record.image_rel).split("/", 1)[0]
        folders = disagreement_folders(prior, int(record.category), algo_reject)
        fired_names = [text.split("  ", 1)[0] for fired, text in rules if fired]
        base = {
            "image_rel": record.image_rel,
            "prior": prior,
            "category": int(record.category),
            "algo": "reject" if algo_reject else "include",
            "fired": "; ".join(fired_names),
            "assign_shift_px": record.assign_shift_px,
            "assign_corr": record.assign_corr,
            "unc_dist_px": record.unc_dist_px,
        }
        rows.append({**base, "folders": ",".join(folders)})
        if not folders:
            continue
        source = os.path.join(sheet_root, str(record.image_rel).replace("/", os.sep))
        sheet = cv2.imread(source, cv2.IMREAD_COLOR)
        if sheet is None:
            raise FileNotFoundError(source)
        human = f"human {int(record.category)}  {CATEGORIES[int(record.category)]}    prior {prior.upper()}"
        for folder in folders:
            painted = annotate_sheet(sheet, TITLES[folder], human, rules)
            destination = os.path.join(out_dir, folder, os.path.basename(str(record.image_rel)))
            if not cv2.imwrite(destination, painted):
                raise OSError(destination)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "algo_reclass.csv"), index=False)
    lines = ["folder\tcount"]
    for folder in FOLDERS:
        count = int(table["folders"].str.contains(folder).sum())
        lines.append(f"{folder}\t{count}")
    lines.append(f"algorithm_reject\t{int((table['algo'] == 'reject').sum())}")
    lines.append(f"algorithm_include\t{int((table['algo'] == 'include').sum())}")
    with open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return table


def main() -> None:
    table = export_disagreements()
    print(open(os.path.join(OUT_DIR, "index.txt"), encoding="utf-8").read())
    print(OUT_DIR)


if __name__ == "__main__":
    main()
