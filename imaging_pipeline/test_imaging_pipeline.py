"""Marker and resume tests. No microscope and no window."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from imaging_pipeline.config import PipelineConfig, load_config, save_config
from imaging_pipeline.gui import list_lines, resume_message, run_self_test
from imaging_pipeline.tasks import (
    build_task_list,
    clear_markers,
    first_incomplete,
    ids_cleared_by_redo,
    ids_cleared_by_restart,
    mark_done,
    read_positions,
    task_status,
    write_spine_count,
)


def _write_csv(folder: Path) -> Path:
    path = folder / "positions.csv"
    path.write_text(
        "pos_id, obj_pos, x_pos_mm, y_pos_mm, z_pos_mm\n"
        "1, 1, 1.0, 2.0, 0.04\n"
        "1, 1, 1.1, 2.0, 0.04\n"
        "2, 1, 3.0, 4.0, 0.05\n"
        "3, 1, 5.0, 6.0, 0.06\n",
        encoding="utf-8",
    )
    return path


def _write_positions(savefolder: Path, slot: str, site_ids: list[int]) -> None:
    export = savefolder / f"{slot}_001"
    export.mkdir(parents=True)
    lines = ["pos_id,x_um,y_um,z_um"]
    lines.extend(f"{site_id},0,0,0" for site_id in site_ids)
    (export / "assigned_relative_um_pos.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


class ImagingPipelineTaskTest(unittest.TestCase):
    def test_repeated_pos_id_gets_the_next_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            positions = read_positions(_write_csv(Path(tmp)))
        self.assertEqual([item.slot for item in positions], ["1_pos1", "1_pos2", "2_pos1", "3_pos1"])

    def test_resume_skips_done_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_csv(root)
            tasks = build_task_list(csv_path, root, top_n=2)
            mark_done(root, tasks[0].task_id)
            nxt = first_incomplete(root, tasks)
            self.assertEqual(nxt.task_id, tasks[1].task_id)
            self.assertIn("Next task: Thin-branch positions 1_pos1", resume_message(root, tasks))

    def test_restart_from_a_later_lowmag_keeps_earlier_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_csv(root)
            _write_positions(root, "1_pos1", [1])
            _write_positions(root, "2_pos1", [1])
            tasks = build_task_list(csv_path, root, top_n=1)
            for task in tasks:
                mark_done(root, task.task_id, output_path=str(root / "keep.tif"))
            image = root / "keep.tif"
            image.write_bytes(b"image")
            selected = next(task for task in tasks if task.task_id == "lowmag-2_pos1")
            cleared = ids_cleared_by_restart(tasks, selected.task_id)
            self.assertIn("lowmag-2_pos1", cleared)
            self.assertIn("uncage-1", cleared)
            self.assertNotIn("lowmag-1_pos1", cleared)
            self.assertNotIn("highmag-1_pos1-site1", cleared)
            clear_markers(root, cleared, "restart")
            self.assertEqual(task_status(root, "lowmag-1_pos1"), "done")
            self.assertEqual(task_status(root, "lowmag-2_pos1"), "pending")
            self.assertEqual(task_status(root, "uncage-1"), "pending")
            self.assertTrue(image.is_file())

    def test_redo_one_lowmag_clears_its_children_and_uncage_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_csv(root)
            _write_positions(root, "1_pos1", [1, 2])
            _write_positions(root, "2_pos1", [1])
            tasks = build_task_list(csv_path, root, top_n=1)
            cleared = ids_cleared_by_redo(tasks, "lowmag-1_pos1")
            self.assertIn("pick-1_pos1", cleared)
            self.assertIn("highmag-1_pos1-site1", cleared)
            self.assertIn("spine-1_pos1-site2", cleared)
            self.assertIn("uncage-1", cleared)
            self.assertNotIn("lowmag-2_pos1", cleared)
            self.assertNotIn("highmag-2_pos1-site1", cleared)

    def test_uncage_label_uses_the_higher_spine_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_csv(root)
            _write_positions(root, "1_pos1", [1])
            _write_positions(root, "2_pos1", [1])
            write_spine_count(root, "spine-1_pos1-site1", 2)
            write_spine_count(root, "spine-2_pos1-site1", 9)
            tasks = build_task_list(csv_path, root, top_n=1)
            uncage = next(task for task in tasks if task.stage == "uncage")
            self.assertIn("2_pos1 site 1", uncage.label)
            self.assertIn("9 spines", uncage.label)

    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = save_config(PipelineConfig(pos_csv=str(root / "p.csv"), savefolder=str(root), top_n=4))
            loaded = load_config(root)
            self.assertEqual(loaded.top_n, 4)
            self.assertEqual(saved.name, "pipeline_config.json")

    def test_headless_self_test_prints_the_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = _write_csv(root)
            os.environ["IMAGING_PIPELINE_SELF_TEST"] = "1"
            os.environ["IMAGING_PIPELINE_POS_CSV"] = str(csv_path)
            os.environ["IMAGING_PIPELINE_SAVEFOLDER"] = str(root)
            os.environ["IMAGING_PIPELINE_TOP_N"] = "1"
            try:
                run_self_test()
            finally:
                os.environ.pop("IMAGING_PIPELINE_SELF_TEST", None)
            self.assertTrue(list_lines(root, build_task_list(csv_path, root, 1)))


if __name__ == "__main__":
    unittest.main()
