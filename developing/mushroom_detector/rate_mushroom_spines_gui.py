# -*- coding: utf-8 -*-
"""
GUI to rate detected mushroom spines (1-4 scale).

Loads PNG paths from mushroom_spine_assign_summary.csv when present, then drops
spines whose heads are within 1 um XY of another spine in the same Z±3 window.

1 = absolutely reject
2 = usually not selected
3 = acceptable
4 = appropriate
N = next (skip without rating; spine stays unrated)
B / Left = previous

Run with deepd3 venv (PyQt5 required), e.g.:
  python rate_mushroom_spines_gui.py "G:\\...\\auto1 - Copy"
"""
from __future__ import annotations

import csv
import datetime
import os
import re
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from respan_mushroom_core import (  # noqa: E402
    DEDUPE_MUSHROOM_XY_SEP_UM,
    MUSHROOM_ASSIGN_SUMMARY_FILENAME,
    RATING_Z_HALF_WINDOW,
    filter_mushroom_rows_by_z_xy_proximity,
)

# --- edit here ---
# Single folder (CLI arg overrides). If None, DEFAULT_ROOT_FOLDERS is used.
DEFAULT_ROOT_FOLDER: str | None = None

# Folders to scan for spine overlay PNGs (training + new auto1 sessions).
DEFAULT_ROOT_FOLDERS = [
    r"G:\ImagingData\Tetsuya\20260515\auto1 - Copy",
    r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend - Copy",
    r"G:\ImagingData\Tetsuya\20260608\mushroom_multi_dend",
    r"G:\ImagingData\Tetsuya\20260530\auto1",
    r"G:\ImagingData\Tetsuya\20260528\auto1",
    r"G:\ImagingData\Tetsuya\20260526\auto1",
    r"G:\ImagingData\Tetsuya\20260510\auto1",
    r"G:\ImagingData\Tetsuya\20260429\auto1",
]

# Combined ratings CSV when rating across multiple folders.
COMBINED_RATINGS_DIR = r"G:\ImagingData\Tetsuya\mushroom_training"
RATINGS_FILENAME = "mushroom_spine_ratings.csv"

RATING_OPTIONS = {
    1: "absolutely_reject",
    2: "usually_not_selected",
    3: "acceptable",
    4: "appropriate",
}
RATING_LABELS = {
    1: "1 Absolutely reject",
    2: "2 Usually not selected",
    3: "3 Acceptable",
    4: "4 Appropriate",
}

SPINE_PNG_RE = re.compile(r".*_\d{3}\.png$", re.IGNORECASE)
EXCLUDE_PNG_SUBSTRINGS = ("_mip", "_roi", "_shaft", "_compare", "_watershed", "_overlay")
EXCLUDE_PATH_SUBSTRINGS = (
    "spine_timeseries",
    "zproj_overlays",
    "z_triplets",
    "respan_runs",
    "validation_data",
    "seg_masks",
)


def is_spine_overlay_png(path: str) -> bool:
    name = os.path.basename(path).lower()
    if not SPINE_PNG_RE.match(name):
        return False
    if any(token in name for token in EXCLUDE_PNG_SUBSTRINGS):
        return False
    path_lower = path.replace("\\", "/").lower()
    return not any(token in path_lower for token in EXCLUDE_PATH_SUBSTRINGS)


def is_direct_mushroom_spine_png(path: str) -> bool:
    """True for {savefolder}/{base_name}_NNN.png (not timeseries / triplet exports)."""
    if not is_spine_overlay_png(path):
        return False
    parent_name = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    suffix = stem[len(parent_name) + 1 :] if stem.startswith(parent_name + "_") else ""
    return len(suffix) == 3 and suffix.isdigit()


def _load_summary_rows(root_folder: str) -> list[dict[str, str]]:
    summary_path = os.path.join(root_folder, MUSHROOM_ASSIGN_SUMMARY_FILENAME)
    if not os.path.isfile(summary_path):
        return []
    with open(summary_path, newline="", encoding="utf-8") as fobj:
        return list(csv.DictReader(fobj))


def collect_spine_png_paths(
    root_folders: list[str],
    *,
    z_half_window: int = RATING_Z_HALF_WINDOW,
    min_xy_sep_um: float = DEDUPE_MUSHROOM_XY_SEP_UM,
) -> list[str]:
    """Collect mushroom spine overlay PNGs for manual rating."""
    paths: list[str] = []
    seen: set[str] = set()

    for root_folder in root_folders:
        if not os.path.isdir(root_folder):
            continue

        summary_rows = _load_summary_rows(root_folder)
        if summary_rows:
            filtered_rows = filter_mushroom_rows_by_z_xy_proximity(
                summary_rows,
                min_xy_sep_um=min_xy_sep_um,
                z_half_window=z_half_window,
            )
            for row in filtered_rows:
                png_path = row.get("png_path", "")
                if not png_path or not os.path.isfile(png_path):
                    continue
                if png_path in seen:
                    continue
                seen.add(png_path)
                paths.append(png_path)
            continue

        for dirpath, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if not filename.lower().endswith(".png"):
                    continue
                full_path = os.path.join(dirpath, filename)
                if not is_direct_mushroom_spine_png(full_path):
                    continue
                if full_path in seen:
                    continue
                seen.add(full_path)
                paths.append(full_path)

    return sorted(
        paths,
        key=lambda p: (
            _root_folder_sort_key(p, root_folders),
            os.path.dirname(p).lower(),
            os.path.basename(p).lower(),
        ),
    )


def _root_folder_sort_key(png_path: str, root_folders: list[str]) -> int:
    for idx, root_folder in enumerate(root_folders):
        try:
            common = os.path.commonpath([png_path, root_folder])
        except ValueError:
            continue
        if common == os.path.normcase(os.path.normpath(root_folder)):
            return idx
    return len(root_folders)


def resolve_root_folders(cli_folder: str | None = None) -> list[str]:
    if cli_folder:
        return [cli_folder]
    if DEFAULT_ROOT_FOLDER:
        return [DEFAULT_ROOT_FOLDER]
    return list(DEFAULT_ROOT_FOLDERS)


def ratings_csv_path(root_folders: list[str]) -> str:
    if len(root_folders) == 1:
        return os.path.join(root_folders[0], RATINGS_FILENAME)
    os.makedirs(COMBINED_RATINGS_DIR, exist_ok=True)
    return os.path.join(COMBINED_RATINGS_DIR, RATINGS_FILENAME)


def load_ratings(csv_path: str) -> dict[str, dict]:
    if not os.path.isfile(csv_path):
        return {}
    ratings: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            png_path = row.get("png_path", "")
            if not png_path:
                continue
            try:
                score = int(row["rating"])
            except (KeyError, ValueError):
                continue
            ratings[png_path] = {
                "rating": score,
                "rating_label": row.get("rating_label", RATING_OPTIONS.get(score, "")),
                "rated_at": row.get("rated_at", ""),
            }
    return ratings


def load_ratings_for_folders(root_folders: list[str]) -> dict[str, dict]:
    """Load per-folder ratings, then override with combined CSV when present."""
    merged: dict[str, dict] = {}
    for root_folder in root_folders:
        folder_csv = os.path.join(root_folder, RATINGS_FILENAME)
        merged.update(load_ratings(folder_csv))
    merged.update(load_ratings(ratings_csv_path(root_folders)))
    return merged


def save_ratings(
    csv_path: str,
    ratings: dict[str, dict],
    ordered_paths: list[str],
    root_folders: list[str] | None = None,
) -> None:
    fieldnames = ["png_path", "source_folder", "rating", "rating_label", "rated_at"]
    rows = []
    seen = set()
    root_folders = root_folders or []

    def _source_folder(png_path: str) -> str:
        for root_folder in root_folders:
            try:
                common = os.path.commonpath([png_path, root_folder])
            except ValueError:
                continue
            if common == os.path.normcase(os.path.normpath(root_folder)):
                return root_folder
        return os.path.dirname(png_path)

    for png_path in ordered_paths:
        if png_path in ratings:
            info = ratings[png_path]
            rows.append({
                "png_path": png_path,
                "source_folder": _source_folder(png_path),
                "rating": info["rating"],
                "rating_label": info["rating_label"],
                "rated_at": info["rated_at"],
            })
            seen.add(png_path)
    for png_path, info in ratings.items():
        if png_path in seen:
            continue
        rows.append({
            "png_path": png_path,
            "source_folder": _source_folder(png_path),
            "rating": info["rating"],
            "rating_label": info["rating_label"],
            "rated_at": info["rated_at"],
        })

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class MushroomSpineRatingGUI(QMainWindow):
    def __init__(self, root_folders: list[str]):
        super().__init__()
        self.root_folders = root_folders
        self.png_paths = collect_spine_png_paths(root_folders)
        self.csv_path = ratings_csv_path(root_folders)
        self.ratings = load_ratings_for_folders(root_folders)
        self.current_index = self._first_unrated_index()

        self.setWindowTitle("Mushroom Spine Rating")
        self.setMinimumSize(1100, 780)
        self._build_ui()
        self._refresh_view()
        self._update_rating_button_styles()

    def _first_unrated_index(self) -> int:
        for idx, path in enumerate(self.png_paths):
            if path not in self.ratings:
                return idx
        return max(0, len(self.png_paths) - 1)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("Mushroom spine rating")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        self.path_label = QLabel()
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.current_rating_label = QLabel()
        self.current_rating_label.setAlignment(Qt.AlignCenter)
        self.current_rating_label.setFont(QFont("Arial", 11))
        layout.addWidget(self.current_rating_label)

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(480)
        self.image_label.setStyleSheet("background-color: #222; color: #ccc;")
        layout.addWidget(self.image_label, stretch=1)

        rating_row = QHBoxLayout()
        self.rating_buttons: dict[int, QPushButton] = {}
        for score in (1, 2, 3, 4):
            btn = QPushButton(RATING_LABELS[score])
            btn.setMinimumHeight(48)
            btn.setFont(QFont("Arial", 11, QFont.Bold))
            btn.clicked.connect(lambda _checked=False, s=score: self._apply_rating(s))
            self.rating_buttons[score] = btn
            rating_row.addWidget(btn)
        layout.addLayout(rating_row)

        nav_row = QHBoxLayout()
        self.back_button = QPushButton("Previous (B / Left)")
        self.back_button.setMinimumHeight(40)
        self.back_button.clicked.connect(self._go_back)
        nav_row.addWidget(self.back_button)

        self.next_button = QPushButton("Next without rating (N)")
        self.next_button.setMinimumHeight(40)
        self.next_button.clicked.connect(self._go_next)
        nav_row.addWidget(self.next_button)

        help_label = QLabel("Shortcut: 1-4 = rate, N = next (skip), B or Left = back")
        help_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        nav_row.addWidget(help_label, stretch=1)
        layout.addLayout(nav_row)

    def _refresh_view(self) -> None:
        total = len(self.png_paths)
        if total == 0:
            self.progress_label.setText("No spine PNG files found.")
            self.path_label.setText("\n".join(self.root_folders))
            self.image_label.setText("No images to rate.")
            self.back_button.setEnabled(False)
            for btn in self.rating_buttons.values():
                btn.setEnabled(False)
            return

        idx = min(max(self.current_index, 0), total - 1)
        self.current_index = idx
        png_path = self.png_paths[idx]
        rated_count = sum(1 for p in self.png_paths if p in self.ratings)

        self.progress_label.setText(
            f"{idx + 1} / {total}   |   rated: {rated_count} / {total}"
        )
        self.path_label.setText(png_path)

        pixmap = QPixmap(png_path)
        if pixmap.isNull():
            self.image_label.setText(f"Failed to load image:\n{png_path}")
            self._pixmap = None
        else:
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
            self._pixmap = pixmap

        if png_path in self.ratings:
            info = self.ratings[png_path]
            self.current_rating_label.setText(
                f"Current rating: {info['rating']} ({RATING_LABELS[info['rating']]})"
            )
        else:
            self.current_rating_label.setText("Current rating: not rated yet")

        self.back_button.setEnabled(idx > 0)
        self._update_rating_button_styles()

        if rated_count == total:
            self.progress_label.setText(
                f"All rated ({total}/{total}). You can go back and revise."
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_pixmap", None) is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)

    def _update_rating_button_styles(self) -> None:
        if not self.png_paths:
            return
        current_path = self.png_paths[self.current_index]
        current_score = self.ratings.get(current_path, {}).get("rating")
        base_style = "font-weight: bold; min-height: 48px;"
        selected_style = base_style + " background-color: #4CAF50; color: white;"
        for score, btn in self.rating_buttons.items():
            btn.setStyleSheet(selected_style if score == current_score else base_style)

    def _apply_rating(self, score: int) -> None:
        if not self.png_paths:
            return
        png_path = self.png_paths[self.current_index]
        self.ratings[png_path] = {
            "rating": score,
            "rating_label": RATING_OPTIONS[score],
            "rated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        save_ratings(self.csv_path, self.ratings, self.png_paths, self.root_folders)

        if self.current_index < len(self.png_paths) - 1:
            self.current_index += 1
        else:
            QMessageBox.information(
                self,
                "Done",
                f"Reached the last image.\nRatings saved to:\n{self.csv_path}",
            )
        self._refresh_view()

    def _go_back(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._refresh_view()

    def _go_next(self) -> None:
        """Advance without saving a rating (N / Next)."""
        if not self.png_paths:
            return
        if self.current_index < len(self.png_paths) - 1:
            self.current_index += 1
        self._refresh_view()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
            self._apply_rating(int(event.text()))
            return
        if key in (Qt.Key_N, Qt.Key_Right):
            self._go_next()
            return
        if key in (Qt.Key_B, Qt.Key_Left):
            self._go_back()
            return
        super().keyPressEvent(event)


def main() -> None:
    cli_folder = sys.argv[1] if len(sys.argv) > 1 else None
    root_folders = resolve_root_folders(cli_folder)

    missing = [folder for folder in root_folders if not os.path.isdir(folder)]
    if missing:
        print("Folder(s) not found:")
        for folder in missing:
            print(f"  {folder}")
        sys.exit(1)

    print("Rating folders:")
    for folder in root_folders:
        print(f"  {folder}")
    print("Ratings CSV:", ratings_csv_path(root_folders))
    png_paths = collect_spine_png_paths(root_folders)
    print(
        f"Spine PNGs for rating: {len(png_paths)} "
        f"(Z±{RATING_Z_HALF_WINDOW}, min XY sep {DEDUPE_MUSHROOM_XY_SEP_UM} um)"
    )

    app = QApplication(sys.argv)
    window = MushroomSpineRatingGUI(root_folders)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
