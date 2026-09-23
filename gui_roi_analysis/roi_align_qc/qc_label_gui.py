"""Review reject and keep sheets and record a QC category.

Left and Right move between images. Keys 0-9, including the numeric keypad,
or a click, set the category. The choice is written immediately. Keep sheets
with no saved choice show category 9. Closing the window remembers the image
so the next launch resumes there.

Usage:
    python qc_label_gui.py
    python qc_label_gui.py --root PATH
"""

from __future__ import annotations

import argparse
import os
import sys

from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import QImage, QKeyEvent, QKeySequence, QPainter, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from qc_assignment_source import SHEET_BANNER_H, AssignmentLookup  # noqa: E402
from qc_label_store import (  # noqa: E402
    CATEGORIES,
    LabelStore,
    default_category,
    list_images,
    start_index,
)

DEFAULT_ROOT = r"//RY-LAB-WS04/ImagingData/Tetsuya/20260701/auto1/roi_align_qc/reject_qc"


def pixmap_from_bgr(image) -> QPixmap:
    """Copy a BGR uint8 image into a pixmap the canvas can scale."""
    height, width = image.shape[:2]
    rgb = image[:, :, ::-1].copy()
    qimage = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def category_from_key(key: int) -> int | None:
    """Map a main-row or keypad digit to a category id."""
    if Qt.Key_0 <= key <= Qt.Key_9:
        return int(key - Qt.Key_0)
    return None


MIN_ZOOM = 0.05
MAX_ZOOM = 12.0
ZOOM_STEP = 1.15


def apply_zoom(
    scale: float,
    origin_x: float,
    origin_y: float,
    cursor_x: float,
    cursor_y: float,
    steps: float,
) -> tuple[float, float, float]:
    """Zoom around the cursor. The image point under the cursor stays put."""
    new_scale = min(MAX_ZOOM, max(MIN_ZOOM, scale * (ZOOM_STEP ** steps)))
    image_x = (cursor_x - origin_x) / scale
    image_y = (cursor_y - origin_y) / scale
    return new_scale, cursor_x - image_x * new_scale, cursor_y - image_y * new_scale


def apply_pan(origin_x: float, origin_y: float, dx: float, dy: float) -> tuple[float, float]:
    """Move the image by a screen-pixel drag."""
    return origin_x + dx, origin_y + dy


def fit_origin(view_w: float, view_h: float, image_w: float, image_h: float, scale: float) -> tuple[float, float]:
    """Top-left of an image centered in the view at ``scale``."""
    return (view_w - image_w * scale) / 2.0, (view_h - image_h * scale) / 2.0


def matched_pre_view(
    sheet_scale: float,
    sheet_origin_y: float,
    view_w: float,
    image_w: float,
) -> tuple[float, float, float]:
    """Scale and origin that line the assignment square up with the first pre square.

    The sheet pixmap starts with a banner, then the pre caption and the 512 px
    square. The assignment pixmap starts with only that caption, so it is shifted
    down by the banner height. It is placed against the right edge of its view.
    """
    origin_x = max(0.0, view_w - image_w * sheet_scale)
    origin_y = sheet_origin_y + SHEET_BANNER_H * sheet_scale
    return sheet_scale, origin_x, origin_y


class ImageCanvas(QWidget):
    """Fitted image. The wheel zooms at the cursor. Dragging pans."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.NoFocus)
        self.setMouseTracking(True)
        self._pixmap = QPixmap()
        self._scale = 1.0
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._drag_last: QPoint | None = None
        self._user_view = False
        self.refit_hook = None

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._drag_last = None
        self._user_view = False
        self._fit()
        self.update()

    def _fit(self) -> None:
        if self._pixmap.isNull() or self.width() < 2 or self.height() < 2:
            self._scale = 1.0
            self._origin_x = 0.0
            self._origin_y = 0.0
            return
        self._scale = min(self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        self._origin_x, self._origin_y = fit_origin(
            self.width(), self.height(), self._pixmap.width(), self._pixmap.height(), self._scale
        )

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._pixmap.isNull():
            return
        painter.translate(self._origin_x, self._origin_y)
        painter.scale(self._scale, self._scale)
        painter.drawPixmap(0, 0, self._pixmap)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._pixmap.isNull():
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        cursor = event.pos()
        self._scale, self._origin_x, self._origin_y = apply_zoom(
            self._scale, self._origin_x, self._origin_y, cursor.x(), cursor.y(), steps
        )
        self._user_view = True
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_last is None:
            return
        pos = event.pos()
        self._origin_x, self._origin_y = apply_pan(
            self._origin_x, self._origin_y, pos.x() - self._drag_last.x(), pos.y() - self._drag_last.y()
        )
        self._drag_last = pos
        self._user_view = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_last = None
            self.setCursor(Qt.ArrowCursor)

    def set_view(self, scale: float, origin_x: float, origin_y: float) -> None:
        """Show the pixmap at a scale chosen to match another canvas."""
        self._scale = scale
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._user_view = False
        self.update()

    def mouseDoubleClickEvent(self, _event) -> None:  # noqa: N802
        self._user_view = False
        if self.refit_hook is not None:
            self.refit_hook()
            return
        self._fit()
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._user_view:
            return
        if self.refit_hook is not None:
            QTimer.singleShot(0, self.refit_hook)
            return
        self._fit()


class QcLabelWindow(QMainWindow):
    """One image, ten category buttons, and a note for category 0."""

    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root
        self.items = list_images(root)
        self.store = LabelStore(root)
        self.assignments = AssignmentLookup()
        rels = [item.rel_path for item in self.items]
        self.index = start_index(len(self.items), set(self.store.records), rels, self.store.last_rel)
        self.setWindowTitle("QC category review")
        self.resize(1200, 800)

        self.status = QLabel("")
        self.assign_canvas = ImageCanvas()
        self.canvas = ImageCanvas()
        sheets = QHBoxLayout()
        sheets.addWidget(self.assign_canvas, stretch=1)
        sheets.addWidget(self.canvas, stretch=2)
        self.assign_canvas.refit_hook = self._sync_assignment
        self.canvas.refit_hook = self._sync_assignment

        self.buttons = QButtonGroup(self)
        self.buttons.setExclusive(True)
        button_row = QHBoxLayout()
        for category in (1, 2, 3, 4, 5, 6, 7, 8, 9, 0):
            button = QPushButton(f"{category}  {CATEGORIES[category]}")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, cat=category: self._choose(cat))
            self.buttons.addButton(button, category)
            button_row.addWidget(button)

        self.note = QLineEdit()
        self.note.setPlaceholderText("Note for category 0")
        self.note.editingFinished.connect(self._save_note)
        self._next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self._next_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._next_shortcut.activated.connect(lambda: self._move(1))
        self._prev_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self._prev_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._prev_shortcut.activated.connect(lambda: self._move(-1))

        layout = QVBoxLayout()
        layout.addWidget(self.status)
        layout.addLayout(sheets, stretch=1)
        layout.addLayout(button_row)
        layout.addWidget(self.note)
        holder = QWidget()
        holder.setLayout(layout)
        self.setCentralWidget(holder)
        self._show_current()

    def _show_current(self) -> None:
        if not self.items:
            self.status.setText("No PNG files in reject/ or keep/")
            return
        item = self.items[self.index]
        self.canvas.set_pixmap(QPixmap(item.abs_path))
        assignment, assignment_note = self.assignments.column_for(item.session, item.group, item.set_label)
        if assignment is None:
            self.assign_canvas.set_pixmap(QPixmap())
        else:
            self.assign_canvas.set_pixmap(pixmap_from_bgr(assignment))
        saved = self.store.record_for(item.rel_path)
        category = saved.category if saved is not None else default_category(item.pool)
        self._mark_button(category)
        self.note.blockSignals(True)
        self.note.setText(saved.note if saved is not None else "")
        self.note.blockSignals(False)
        state = "saved" if saved is not None else "not saved"
        shown = "-" if category is None else str(category)
        self.status.setText(
            f"{self.index + 1} / {len(self.items)}    {item.original_label}    "
            f"{item.rel_path}    category {shown} ({state})    assignment {assignment_note}"
        )
        self.store.save_last(item.rel_path)
        QTimer.singleShot(0, self._sync_assignment)

    def _sync_assignment(self) -> None:
        """Draw the assignment max projection at the same size as the first pre."""
        assign = self.assign_canvas
        sheet = self.canvas
        if sheet._pixmap.isNull() or sheet.width() < 2:
            return
        if not sheet._user_view:
            sheet._fit()
        if assign._user_view or assign._pixmap.isNull() or assign.width() < 2:
            return
        scale, origin_x, origin_y = matched_pre_view(
            sheet._scale, sheet._origin_y, assign.width(), assign._pixmap.width()
        )
        assign.set_view(scale, origin_x, origin_y)

    def _mark_button(self, category: int | None) -> None:
        if category is None:
            checked = self.buttons.checkedButton()
            if checked is not None:
                self.buttons.setExclusive(False)
                checked.setChecked(False)
                self.buttons.setExclusive(True)
            return
        button = self.buttons.button(category)
        if button is not None:
            button.setChecked(True)

    def _choose(self, category: int) -> None:
        if not self.items:
            return
        item = self.items[self.index]
        note = self.note.text() if category == 0 else ""
        self.store.save_label(item, category, note)
        self._show_current()

    def _save_note(self) -> None:
        if not self.items:
            return
        item = self.items[self.index]
        saved = self.store.record_for(item.rel_path)
        if saved is None or saved.category != 0:
            return
        if saved.note == self.note.text().strip():
            return
        self.store.save_label(item, 0, self.note.text())
        self._show_current()

    def _move(self, step: int) -> None:
        if not self.items:
            return
        self._persist_keep_default()
        self.index = (self.index + step) % len(self.items)
        self._show_current()

    def _persist_keep_default(self) -> None:
        """Leaving a keep sheet with no explicit choice records category 9."""
        item = self.items[self.index]
        if self.store.record_for(item.rel_path) is not None:
            return
        if item.pool != "keep":
            return
        self.store.save_label(item, 9, "")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        category = category_from_key(event.key())
        if category is not None and not isinstance(self.focusWidget(), QLineEdit):
            self._choose(category)
            return
        super().keyPressEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label reject and keep QC sheets.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Folder that contains reject/ and keep/")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    window = QcLabelWindow(args.root)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
