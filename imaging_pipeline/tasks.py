"""Build the day's task list and decide which completion markers to keep."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_POS_LINE = re.compile(
    r"^[A-Za-z0-9_]{1,100},\s*\d+,\s*-?\d+(?:\.\d+)?,\s*-?\d+(?:\.\d+)?,\s*-?\d+(?:\.\d+)?$"
)
_HEADER = "pos_id, obj_pos, x_pos_mm, y_pos_mm, z_pos_mm"
STATE_DIR_NAME = "pipeline_state"
JOURNAL_NAME = "pipeline_journal.csv"


@dataclass(frozen=True)
class CsvPosition:
    pos_id: str
    nth: int
    x_mm: float
    y_mm: float
    z_mm: float

    @property
    def slot(self) -> str:
        return f"{self.pos_id}_pos{self.nth}"


@dataclass(frozen=True)
class Task:
    task_id: str
    stage: str
    label: str
    order: int
    parent_id: str | None
    lowmag_id: str | None


def read_positions(csv_path: Path) -> list[CsvPosition]:
    """Read a stage-position CSV. Same columns as the existing position files."""
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != _HEADER:
        raise ValueError(f"CSV must start with: {_HEADER}")
    positions: list[CsvPosition] = []
    previous_id = ""
    nth = 0
    for line in lines[1:]:
        if not line.strip() or not _POS_LINE.match(line):
            continue
        pos_id = line.split(",")[0].strip()
        tail = line[line.find(",") + 1 :]
        x_mm, y_mm, z_mm = (float(part) for part in tail.split(",")[1:4])
        if pos_id == previous_id:
            nth += 1
        else:
            nth = 1
            previous_id = pos_id
        positions.append(CsvPosition(pos_id, nth, x_mm, y_mm, z_mm))
    return positions


def state_dir(savefolder: Path) -> Path:
    return savefolder / STATE_DIR_NAME


def marker_path(savefolder: Path, task_id: str, kind: str) -> Path:
    return state_dir(savefolder) / f"{task_id}.{kind}"


def task_status(savefolder: Path, task_id: str) -> str:
    if marker_path(savefolder, task_id, "done").is_file():
        return "done"
    if marker_path(savefolder, task_id, "failed").is_file():
        return "failed"
    if marker_path(savefolder, task_id, "running").is_file():
        return "running"
    return "pending"


def _append_journal(savefolder: Path, action: str, task_id: str, detail: str) -> None:
    folder = state_dir(savefolder)
    folder.mkdir(parents=True, exist_ok=True)
    path = savefolder / JOURNAL_NAME
    if not path.is_file():
        path.write_text("time_iso,action,task_id,detail\n", encoding="utf-8")
    stamp = datetime.now().isoformat(timespec="seconds")
    safe_detail = detail.replace('"', "'")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f'{stamp},{action},{task_id},"{safe_detail}"\n')


def mark_done(savefolder: Path, task_id: str, output_path: str = "") -> None:
    folder = state_dir(savefolder)
    folder.mkdir(parents=True, exist_ok=True)
    marker_path(savefolder, task_id, "failed").unlink(missing_ok=True)
    marker_path(savefolder, task_id, "running").unlink(missing_ok=True)
    marker_path(savefolder, task_id, "done").write_text(
        f"{datetime.now().isoformat(timespec='seconds')}\n{output_path}\n",
        encoding="utf-8",
    )
    _append_journal(savefolder, "done", task_id, output_path)


def mark_failed(savefolder: Path, task_id: str, detail: str) -> None:
    folder = state_dir(savefolder)
    folder.mkdir(parents=True, exist_ok=True)
    marker_path(savefolder, task_id, "running").unlink(missing_ok=True)
    marker_path(savefolder, task_id, "failed").write_text(detail + "\n", encoding="utf-8")
    _append_journal(savefolder, "failed", task_id, detail)


def clear_markers(savefolder: Path, task_ids: list[str], action: str) -> None:
    """Remove completion markers. Does not delete acquired images."""
    for task_id in task_ids:
        removed = False
        for kind in ("done", "failed", "running"):
            path = marker_path(savefolder, task_id, kind)
            if path.is_file():
                path.unlink()
                removed = True
        if removed:
            _append_journal(savefolder, action, task_id, "marker cleared")


def _um_position_ids(csv_path: Path) -> list[str]:
    ids: list[str] = []
    for index, line in enumerate(csv_path.read_text(encoding="utf-8").splitlines()):
        if index == 0 or not line.strip():
            continue
        ids.append(line.split(",")[0].strip())
    return ids


def assigned_um_csv(savefolder: Path, slot: str) -> Path | None:
    """Latest click-style folder for this low-mag slot, if positions were saved."""
    if not savefolder.is_dir():
        return None
    prefix = f"{slot}_"
    found = [
        child / "assigned_relative_um_pos.csv"
        for child in savefolder.iterdir()
        if child.is_dir() and child.name.startswith(prefix)
        and (child / "assigned_relative_um_pos.csv").is_file()
    ]
    if not found:
        return None
    return max(found, key=lambda path: path.parent.name)


def read_spine_count(savefolder: Path, spine_task_id: str) -> int | None:
    path = state_dir(savefolder) / f"{spine_task_id}.spine_count.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return int(text)


def write_spine_count(savefolder: Path, spine_task_id: str, count: int) -> None:
    folder = state_dir(savefolder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{spine_task_id}.spine_count.txt").write_text(f"{count}\n", encoding="utf-8")


def build_task_list(csv_path: Path, savefolder: Path, top_n: int) -> list[Task]:
    """Low-mag rows from the CSV, high-mag rows from saved position files, then uncage slots."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    tasks: list[Task] = []
    order = 0
    highmag_for_rank: list[tuple[str, str, int | None]] = []

    def add(task_id: str, stage: str, label: str, parent_id: str | None, lowmag_id: str | None) -> None:
        nonlocal order
        tasks.append(Task(task_id, stage, label, order, parent_id, lowmag_id))
        order += 1

    for position in read_positions(csv_path):
        lowmag_id = f"lowmag-{position.slot}"
        add(lowmag_id, "lowmag", f"Low mag {position.slot}", None, lowmag_id)
        pick_id = f"pick-{position.slot}"
        add(pick_id, "pick", f"Thin-branch positions {position.slot}", lowmag_id, lowmag_id)
        um_csv = assigned_um_csv(savefolder, position.slot)
        if um_csv is None:
            continue
        for site_id in _um_position_ids(um_csv):
            high_id = f"highmag-{position.slot}-site{site_id}"
            add(
                high_id,
                "highmag",
                f"High mag {position.slot} site {site_id}",
                lowmag_id,
                lowmag_id,
            )
            spine_id = f"spine-{position.slot}-site{site_id}"
            count = read_spine_count(savefolder, spine_id)
            count_text = "unscored" if count is None else f"{count} spines"
            add(
                spine_id,
                "spine",
                f"Spine finder {position.slot} site {site_id} ({count_text})",
                high_id,
                lowmag_id,
            )
            highmag_for_rank.append((high_id, f"{position.slot} site {site_id}", count))

    ranked = sorted(
        highmag_for_rank,
        key=lambda item: (item[2] is None, -(item[2] or 0), item[1]),
    )
    for rank in range(1, top_n + 1):
        if rank <= len(ranked) and ranked[rank - 1][2] is not None:
            label = f"Pre/Unc/Post rank {rank}: {ranked[rank - 1][1]} ({ranked[rank - 1][2]} spines)"
        else:
            label = f"Pre/Unc/Post rank {rank}: waiting for spine counts"
        add(f"uncage-{rank}", "uncage", label, None, None)
    return tasks


def first_incomplete(savefolder: Path, tasks: list[Task]) -> Task | None:
    for task in tasks:
        if task_status(savefolder, task.task_id) != "done":
            return task
    return None


def ids_cleared_by_restart(tasks: list[Task], selected_id: str) -> list[str]:
    selected = _require(tasks, selected_id)
    return [task.task_id for task in tasks if task.order >= selected.order]


def ids_cleared_by_redo(tasks: list[Task], selected_id: str) -> list[str]:
    """Clear one row, its children, and uncage slots when ranking could change."""
    selected = _require(tasks, selected_id)
    children = _descendant_ids(tasks, selected.task_id)
    cleared = [selected.task_id, *children]
    if selected.stage != "uncage":
        cleared.extend(task.task_id for task in tasks if task.stage == "uncage")
    return list(dict.fromkeys(cleared))


def _require(tasks: list[Task], task_id: str) -> Task:
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(task_id)


def _descendant_ids(tasks: list[Task], root_id: str) -> list[str]:
    by_parent: dict[str, list[str]] = {}
    for task in tasks:
        if task.parent_id:
            by_parent.setdefault(task.parent_id, []).append(task.task_id)
    found: list[str] = []
    stack = list(by_parent.get(root_id, []))
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(by_parent.get(current, []))
    return found


def preview_labels(tasks: list[Task], task_ids: list[str]) -> list[str]:
    wanted = set(task_ids)
    return [task.label for task in tasks if task.task_id in wanted]
