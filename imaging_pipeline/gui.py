"""One window for the day's imaging tasks.

Acquisition is not started from here yet. Resume reports the next unfinished
task. Restart and redo only clear completion markers after confirmation.
"""

from __future__ import annotations

import os
from pathlib import Path

from imaging_pipeline.config import PipelineConfig, load_config, save_config
from imaging_pipeline.tasks import (
    build_task_list,
    clear_markers,
    first_incomplete,
    ids_cleared_by_redo,
    ids_cleared_by_restart,
    preview_labels,
    task_status,
)

_SELF_TEST_ENV = "IMAGING_PIPELINE_SELF_TEST"


def resume_message(savefolder: Path, tasks: list) -> str:
    nxt = first_incomplete(savefolder, tasks)
    if nxt is None:
        return "All listed tasks are done."
    return f"Next task: {nxt.label}. Acquisition is not connected yet."


def tasks_or_error(config: PipelineConfig) -> tuple[list, str]:
    csv_path = Path(config.pos_csv)
    savefolder = Path(config.savefolder)
    if not csv_path.is_file():
        return [], "Choose a position CSV."
    if not str(config.savefolder):
        return [], "Choose a save folder."
    try:
        config.validate()
        return build_task_list(csv_path, savefolder, config.top_n), ""
    except ValueError as exc:
        return [], str(exc)


def list_lines(savefolder: Path, tasks: list) -> list[str]:
    return [f"[{task_status(savefolder, task.task_id)}] {task.label}" for task in tasks]


def run_self_test() -> None:
    """Print the task list and exit. Used when no window should open."""
    config = PipelineConfig(
        pos_csv=os.environ.get("IMAGING_PIPELINE_POS_CSV", ""),
        savefolder=os.environ.get("IMAGING_PIPELINE_SAVEFOLDER", ""),
        top_n=int(os.environ.get("IMAGING_PIPELINE_TOP_N", "8")),
    )
    tasks, error = tasks_or_error(config)
    if error:
        raise SystemExit(error)
    for line in list_lines(Path(config.savefolder), tasks):
        print(line)
    print(resume_message(Path(config.savefolder), tasks))


def build_layout(config: PipelineConfig, lines: list[str]) -> list:
    import PySimpleGUI as sg

    return [
        [sg.Text("Imaging pipeline", font="Arial 14")],
        [
            sg.Text("Position CSV", size=(16, 1)),
            sg.Input(config.pos_csv, key="-CSV-", size=(70, 1)),
            sg.FileBrowse(file_types=(("CSV", "*.csv"),)),
        ],
        [
            sg.Text("Save folder", size=(16, 1)),
            sg.Input(config.savefolder, key="-FOLDER-", size=(70, 1)),
            sg.FolderBrowse(),
        ],
        [
            sg.Text("Top N", size=(16, 1)),
            sg.Input(str(config.top_n), key="-TOPN-", size=(6, 1)),
            sg.Text("Min auto rating"),
            sg.Input(str(config.min_auto_rating), key="-RATING-", size=(6, 1)),
            sg.Text("High-mag zoom"),
            sg.Input(str(config.zoom_highmag), key="-ZOOM-", size=(6, 1)),
            sg.Text("High-mag power %"),
            sg.Input(str(config.highmag_power_percent), key="-POWER-", size=(6, 1)),
        ],
        [
            sg.Text("High-mag setting", size=(16, 1)),
            sg.Input(config.highmag_setting_path, key="-HIGHMAG-", size=(70, 1)),
            sg.FileBrowse(file_types=(("Text", "*.txt"), ("All", "*.*"))),
        ],
        [
            sg.Button("Refresh"),
            sg.Button("Save settings"),
            sg.Button("Resume"),
            sg.Button("Restart from here"),
            sg.Button("Redo this row"),
            sg.Button("Stop"),
        ],
        [
            sg.Listbox(
                lines,
                size=(110, 18),
                key="-TASKS-",
                select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
            )
        ],
        [sg.Multiline(size=(110, 8), key="-LOG-", autoscroll=True, disabled=True)],
    ]


def _config_from_values(values: dict) -> PipelineConfig:
    return PipelineConfig(
        pos_csv=values["-CSV-"].strip(),
        savefolder=values["-FOLDER-"].strip(),
        top_n=int(values["-TOPN-"]),
        min_auto_rating=int(values["-RATING-"]),
        highmag_setting_path=values["-HIGHMAG-"].strip(),
        zoom_highmag=int(values["-ZOOM-"]),
        highmag_power_percent=int(values["-POWER-"]),
    )


def _selected_task(values: dict, tasks: list):
    selected = values["-TASKS-"]
    if not selected:
        return None
    index = None
    lines = list_lines(Path(values["-FOLDER-"].strip()), tasks) if tasks else []
    for index, line in enumerate(lines):
        if line == selected[0]:
            return tasks[index]
    return None


def _confirm_clear(labels: list[str]) -> bool:
    import PySimpleGUI as sg

    if not labels:
        sg.popup("Nothing to clear.")
        return False
    text = "These rows will return to pending. Image files stay on disk.\n\n" + "\n".join(labels)
    return sg.popup_yes_no(text, title="Clear completion markers") == "Yes"


def run_window(initial_folder: str = "") -> None:
    import PySimpleGUI as sg

    sg.theme("DarkBlue3")
    config = load_config(Path(initial_folder)) if initial_folder else PipelineConfig()
    tasks, error = tasks_or_error(config)
    lines = list_lines(Path(config.savefolder), tasks) if not error and config.savefolder else []
    window = sg.Window("Imaging pipeline", build_layout(config, lines), finalize=True)
    log_lines: list[str] = []
    if error and config.pos_csv:
        log_lines.append(error)
    stop_requested = False

    def log(message: str) -> None:
        log_lines.append(message)
        window["-LOG-"].update("\n".join(log_lines[-200:]))

    def reload_from_values(values: dict) -> PipelineConfig | None:
        nonlocal tasks
        try:
            config = _config_from_values(values)
        except ValueError as exc:
            log(str(exc))
            return None
        loaded, error = tasks_or_error(config)
        if error:
            log(error)
            tasks = []
            window["-TASKS-"].update([])
            return None
        tasks = loaded
        window["-TASKS-"].update(list_lines(Path(config.savefolder), tasks))
        return config

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, None):
            break
        if event == "Stop":
            stop_requested = True
            log("Stop requested. The current step is not running yet, so nothing was interrupted.")
            continue
        if event == "Refresh":
            reload_from_values(values)
            continue
        if event == "Save settings":
            config = reload_from_values(values)
            if config is None:
                continue
            path = save_config(config)
            log(f"Saved settings: {path}")
            continue
        if event == "Resume":
            config = reload_from_values(values)
            if config is None:
                continue
            if stop_requested:
                log("Resume cancelled because Stop was pressed. Press Resume again to continue.")
                stop_requested = False
                continue
            log(resume_message(Path(config.savefolder), tasks))
            continue
        if event in ("Restart from here", "Redo this row"):
            config = reload_from_values(values)
            if config is None:
                continue
            selected = _selected_task(values, tasks)
            if selected is None:
                log("Select a row first.")
                continue
            if event == "Restart from here":
                ids = ids_cleared_by_restart(tasks, selected.task_id)
                action = "restart"
            else:
                ids = ids_cleared_by_redo(tasks, selected.task_id)
                action = "redo"
            if not _confirm_clear(preview_labels(tasks, ids)):
                log("Clear cancelled.")
                continue
            clear_markers(Path(config.savefolder), ids, action)
            reload_from_values(values)
            log(f"Cleared {len(ids)} marker(s). Image files were not deleted.")
    window.close()


def main() -> None:
    if os.environ.get(_SELF_TEST_ENV) == "1":
        run_self_test()
        return
    run_window()
