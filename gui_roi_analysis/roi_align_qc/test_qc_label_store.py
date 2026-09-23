"""Saved QC categories survive a restart and keep sheets default to Keep."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from qc_label_gui import (  # noqa: E402
    MAX_ZOOM,
    QcLabelWindow,
    apply_pan,
    apply_zoom,
    category_from_key,
    fit_origin,
)
import qc_label_store  # noqa: E402
from qc_label_store import (  # noqa: E402
    LabelStore,
    default_category,
    list_images,
    start_index,
)


def _write_png(path: str) -> None:
    image = QImage(8, 8, QImage.Format_RGB32)
    image.fill(0xFF202020)
    image.save(path)


def _sample_tree(root: str) -> None:
    os.makedirs(os.path.join(root, "reject"))
    os.makedirs(os.path.join(root, "keep"))
    _write_png(os.path.join(root, "reject", "20260623_a_set0.png"))
    _write_png(os.path.join(root, "keep", "20260623_b_set1.png"))
    with open(os.path.join(root, "index.txt"), "w", encoding="utf-8") as handle:
        handle.write("session\tgroup\tset\tlabel\tstatus\tpath\n")
        handle.write(
            "20260623\ta_\t0.0\tREJECT\tok\t"
            + os.path.join(root, "reject", "20260623_a_set0.png")
            + "\n"
        )
        handle.write(
            "20260623\tb_\t1.0\tKEEP\tok\t"
            + os.path.join(root, "keep", "20260623_b_set1.png")
            + "\n"
        )


class LabelStoreTest(unittest.TestCase):
    def test_order_metadata_and_keep_default(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            _sample_tree(root)
            items = list_images(root)
            self.assertEqual([item.pool for item in items], ["reject", "keep"])
            self.assertEqual(items[0].group, "a_")
            self.assertEqual(items[0].original_label, "REJECT")
            self.assertIsNone(default_category("reject"))
            self.assertEqual(default_category("keep"), 9)

    def test_save_reload_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            _sample_tree(root)
            items = list_images(root)
            store = LabelStore(root)
            store.save_label(items[0], 8, "")
            store.save_label(items[1], 0, "dim field")
            store.save_last(items[1].rel_path)
            again = LabelStore(root)
            self.assertEqual(again.record_for(items[0].rel_path).category, 8)
            self.assertEqual(again.record_for(items[1].rel_path).note, "dim field")
            self.assertEqual(again.last_rel, items[1].rel_path)
            rels = [item.rel_path for item in items]
            self.assertEqual(start_index(2, set(again.records), rels, again.last_rel), 1)
            self.assertEqual(start_index(2, set(), rels, None), 0)

    def test_resume_file_is_kept_when_replace_is_denied(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            _sample_tree(root)
            store = LabelStore(root)
            store.save_last("reject/first.png")
            real_replace = os.replace
            calls = {"n": 0}

            def deny_once(src: str, dst: str) -> None:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise PermissionError(5, "Access is denied")
                real_replace(src, dst)

            original = qc_label_store.os.replace
            qc_label_store.os.replace = deny_once
            try:
                store.save_last("reject/second.png")
            finally:
                qc_label_store.os.replace = original
            again = LabelStore(root)
            self.assertEqual(again.last_rel, "reject/second.png")

    def test_csv_has_the_category_name(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            _sample_tree(root)
            item = list_images(root)[0]
            LabelStore(root).save_label(item, 7, "")
            with open(os.path.join(root, "qc_labels.csv"), encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["category"], "7")
            self.assertIn("Drift", row["category_name"])
            self.assertEqual(row["session"], "20260623")


class ZoomPanTest(unittest.TestCase):
    def test_zoom_keeps_the_cursor_point_fixed(self) -> None:
        scale, origin_x, origin_y = apply_zoom(1.0, 10.0, 20.0, 110.0, 220.0, 1.0)
        self.assertGreater(scale, 1.0)
        self.assertAlmostEqual((110.0 - origin_x) / scale, 100.0)
        self.assertAlmostEqual((220.0 - origin_y) / scale, 200.0)

    def test_zoom_is_clamped_and_pan_adds_the_drag(self) -> None:
        scale, _x, _y = apply_zoom(11.0, 0, 0, 0, 0, 5)
        self.assertEqual(scale, MAX_ZOOM)
        self.assertEqual(apply_pan(3, 4, -2, 5), (1, 9))
        self.assertEqual(fit_origin(200, 100, 100, 50, 1.0), (50.0, 25.0))


class GuiChoiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_digit_keys_and_keep_saved_on_leave(self) -> None:
        self.assertEqual(category_from_key(Qt.Key_0), 0)
        self.assertEqual(category_from_key(Qt.Key_8), 8)
        self.assertIsNone(category_from_key(Qt.Key_Left))
        with tempfile.TemporaryDirectory() as root:
            _sample_tree(root)
            window = QcLabelWindow(root)
            window._choose(4)
            window._move(1)
            window._move(1)
            reloaded = LabelStore(root)
            items = list_images(root)
            self.assertEqual(reloaded.record_for(items[0].rel_path).category, 4)
            self.assertEqual(reloaded.record_for(items[1].rel_path).category, 9)
            self.assertEqual(reloaded.last_rel, items[0].rel_path)


if __name__ == "__main__":
    unittest.main()
