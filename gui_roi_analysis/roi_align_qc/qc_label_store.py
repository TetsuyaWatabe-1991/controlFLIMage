"""Persist manual QC categories for reject and keep preview sheets.

Category ids match the review keypad:

1. ROI is off the spine
2. Uncaging XY is off; the ROI looks correct
3. Uncaging Z is wrong
4. An unrelated object from another Z overlaps the spine
5. A neighboring spine intrudes
6. The first pre frame is far from the intended field
7. Drift is too large to track the same spine
8. The cell is dead or clearly unhealthy
9. Keep (a Reject label would be a mistake)
0. Other reject reason, with a free-text note
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timezone

CATEGORIES: dict[int, str] = {
    1: "ROI is off the spine",
    2: "Uncaging XY is off; ROI looks correct",
    3: "Uncaging Z is wrong",
    4: "Unrelated object from another Z overlaps the spine",
    5: "A neighboring spine intrudes",
    6: "First pre frame is far from the intended field",
    7: "Drift is too large to track the same spine",
    8: "Dead or clearly unhealthy cell",
    9: "Keep",
    0: "Other reject reason",
}

KEEP_DEFAULT_CATEGORY = 9
CSV_NAME = "qc_labels.csv"
STATE_NAME = "qc_label_state.json"
COLUMNS = (
    "image_rel",
    "pool",
    "session",
    "group",
    "set_label",
    "original_label",
    "category",
    "category_name",
    "note",
    "updated_utc",
)


@dataclass(frozen=True)
class ImageItem:
    """One preview PNG and the set it came from."""

    rel_path: str
    abs_path: str
    pool: str
    session: str
    group: str
    set_label: str
    original_label: str


@dataclass(frozen=True)
class LabelRecord:
    """A saved category for one image."""

    category: int
    note: str
    updated_utc: str


def category_name(category: int) -> str:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    return CATEGORIES[category]


def default_category(pool: str) -> int | None:
    """Keep sheets start on Keep. Reject sheets start with no choice."""
    if pool == "keep":
        return KEEP_DEFAULT_CATEGORY
    return None


def start_index(count: int, labeled: set[str], rel_paths: list[str], last_rel: str | None) -> int:
    """Resume at the last viewed image, otherwise the first unlabeled one."""
    if count <= 0:
        return 0
    if last_rel in rel_paths:
        return rel_paths.index(last_rel)
    for index, rel in enumerate(rel_paths):
        if rel not in labeled:
            return index
    return 0


def list_images(root: str) -> list[ImageItem]:
    """Reject sheets first, then keep sheets, each sorted by filename."""
    meta = _index_by_basename(os.path.join(root, "index.txt"))
    items: list[ImageItem] = []
    for pool in ("reject", "keep"):
        folder = os.path.join(root, pool)
        if not os.path.isdir(folder):
            continue
        names = sorted(name for name in os.listdir(folder) if name.lower().endswith(".png"))
        for name in names:
            info = meta.get(name, {})
            items.append(
                ImageItem(
                    rel_path=f"{pool}/{name}",
                    abs_path=os.path.join(folder, name),
                    pool=pool,
                    session=info.get("session", ""),
                    group=info.get("group", ""),
                    set_label=info.get("set_label", ""),
                    original_label=info.get("original_label", pool.upper()),
                )
            )
    return items


def _index_by_basename(path: str) -> dict[str, dict[str, str]]:
    if not os.path.isfile(path):
        return {}
    found: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            image_path = (row.get("path") or "").strip()
            if not image_path:
                continue
            found[os.path.basename(image_path)] = {
                "session": (row.get("session") or "").strip(),
                "group": (row.get("group") or "").strip(),
                "set_label": (row.get("set") or "").strip(),
                "original_label": (row.get("label") or "").strip(),
            }
    return found


class LabelStore:
    """CSV of categories plus the last viewed image."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.csv_path = os.path.join(root, CSV_NAME)
        self.state_path = os.path.join(root, STATE_NAME)
        self.records: dict[str, LabelRecord] = {}
        self.last_rel: str | None = None
        self.load()

    def load(self) -> None:
        self.records = {}
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rel = (row.get("image_rel") or "").strip()
                    raw = (row.get("category") or "").strip()
                    if not rel or raw == "":
                        continue
                    self.records[rel] = LabelRecord(
                        category=int(raw),
                        note=(row.get("note") or "").strip(),
                        updated_utc=(row.get("updated_utc") or "").strip(),
                    )
        self.last_rel = None
        if os.path.isfile(self.state_path):
            with open(self.state_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            last = payload.get("last_rel")
            if isinstance(last, str) and last:
                self.last_rel = last

    def record_for(self, rel_path: str) -> LabelRecord | None:
        return self.records.get(rel_path)

    def save_label(self, item: ImageItem, category: int, note: str) -> LabelRecord:
        record = LabelRecord(
            category=int(category),
            note=note.strip(),
            updated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self.records[item.rel_path] = record
        self._write_csv(list_images(self.root))
        return record

    def save_last(self, rel_path: str) -> None:
        self.last_rel = rel_path
        os.makedirs(self.root, exist_ok=True)
        temporary = self.state_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"last_rel": rel_path}, handle)
        _replace_file(temporary, self.state_path)

    def _write_csv(self, items: list[ImageItem]) -> None:
        by_rel = {item.rel_path: item for item in items}
        os.makedirs(self.root, exist_ok=True)
        temporary = self.csv_path + ".tmp"
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for rel in sorted(self.records):
                record = self.records[rel]
                item = by_rel.get(rel)
                writer.writerow(
                    {
                        "image_rel": rel,
                        "pool": item.pool if item else rel.split("/", 1)[0],
                        "session": item.session if item else "",
                        "group": item.group if item else "",
                        "set_label": item.set_label if item else "",
                        "original_label": item.original_label if item else "",
                        "category": record.category,
                        "category_name": category_name(record.category),
                        "note": record.note,
                        "updated_utc": record.updated_utc,
                    }
                )
        _replace_file(temporary, self.csv_path)


def _replace_file(temporary: str, destination: str) -> None:
    """Replace destination with temporary.

    On a Windows file share, replacing a file that was just written often
    raises PermissionError even though creating the temp file succeeded.
    """
    try:
        os.replace(temporary, destination)
        return
    except PermissionError:
        pass
    try:
        if os.path.isfile(destination):
            os.chmod(destination, stat.S_IWRITE | stat.S_IREAD)
            os.remove(destination)
        os.replace(temporary, destination)
        return
    except PermissionError:
        pass
    shutil.copyfile(temporary, destination)
    try:
        os.remove(temporary)
    except OSError:
        pass
