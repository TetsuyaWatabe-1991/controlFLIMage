"""Set-by-set review viewer for the RESPAN ROI workflow (analayze_all_flim_roi_respan.py).

Shows set_review_panel for one set at a time (spine crops with Spine / Shaft ROIs for
all pre and post frames and the first / last uncaging frame, plus the normalized
quantification). Intended flow: quantify every set with the pre-filled RESPAN ROIs
first, then look through the sets here and only edit the ones that need it. The viewer
can also be opened on its own on an existing combined_df pkl (sets that have no
quantification yet are quantified when they are shown).

Keys:
    Left / Right   previous / next set
    E  (or Enter)  edit the ROIs of this set in the existing ROI GUI (Spine, then Shaft);
                   afterwards this set is re-quantified and the display is updated
    R              toggle reject (same flag file and pkl column as the ROI table GUI).
                   Rejected sets stay in the list and keep their images and data, so a
                   mistaken reject can simply be toggled back.
    Q  (or Esc)    close

The uncaging frames show the uncaging position as a yellow cross on the same tissue as
FLIMage's cross on the raw uncaging image (gui_roi_respan_seg_masks.uncaging_xy_in_tiff;
not corrected towards the spine); the ROI GUI opened with E shows the
same position.

Re-quantification uses the same functions and settings as the workflow
(save_drift_corrected_roi_masks + quantify_intensity_from_flim with the image 20th
percentile as background) and replaces only this set's rows in the all-frames CSV.
The CSV is backed up once per viewer session (<csv>.bak_<timestamp>) before the first
change.

Usage:
    python set_review_viewer.py                      # choose the pkl in a dialog
    python set_review_viewer.py --pkl <combined_df_respan.pkl> [--csv <..._all_frames.csv>]
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.path.dirname(HERE) not in sys.path:
    sys.path.append(os.path.dirname(HERE))

from set_review_panel import draw_set_review, load_set_review  # noqa: E402

CSV_SUFFIX = "_intensity_lifetime_all_frames.csv"


# --------------------------------------------------------------------------- data helpers
def default_csv_for_pkl(pkl_path: str) -> str:
    """Same naming as run_tiff_uncaging_roi_respan (out_csv = pkl -> *_all_frames.csv)."""
    return pkl_path.replace(".pkl", CSV_SUFFIX)


def list_sets(combined_df: pd.DataFrame) -> list[tuple[str, float, str]]:
    """(group, set_label, tiff_path) of every set with a full-size TIFF, in dataframe order."""
    col = "after_align_full_save_path" if "after_align_full_save_path" in combined_df.columns else "after_align_save_path"
    out, seen = [], set()
    for r in combined_df[combined_df.nth_set_label != -1].itertuples():
        key = (str(r.group), float(r.nth_set_label))
        tiff = getattr(r, col)
        if key in seen or not isinstance(tiff, str) or not os.path.exists(tiff):
            continue
        seen.add(key)
        out.append((key[0], key[1], tiff))
    return out


def set_rows(df: pd.DataFrame, group: str, set_label: float, *, csv_like: bool = False) -> pd.Series:
    sl = "set_label" if csv_like else "nth_set_label"
    if not len(df) or sl not in df.columns:
        return pd.Series(False, index=df.index)
    return (df["group"].astype(str) == str(group)) & (pd.to_numeric(df[sl], errors="coerce") == float(set_label))


def replace_set_rows(all_q: pd.DataFrame, new_q: pd.DataFrame | None, group: str, set_label: float) -> pd.DataFrame:
    """Replace one set's rows (keeps the position of the set; None/empty removes it)."""
    m = set_rows(all_q, group, set_label, csv_like=True)
    new_q = new_q if new_q is not None and len(new_q) else None
    if not m.any():
        return all_q if new_q is None else pd.concat([all_q, new_q], ignore_index=True)
    first = int(np.flatnonzero(m.to_numpy())[0])
    before, after = all_q.iloc[:first][~m.iloc[:first]], all_q.iloc[first:][~m.iloc[first:]]
    parts = [before] + ([new_q] if new_q is not None else []) + [after]
    return pd.concat(parts, ignore_index=True)


def _atomic_replace(tmp: str, path: str) -> None:
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # file open in another program (e.g. Excel)
            time.sleep(0.2)
    raise PermissionError(f"could not replace {path} (open in another program?)")


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp_{os.getpid()}"
    df.to_csv(tmp, index=False)
    _atomic_replace(tmp, path)


def atomic_write_pickle(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp_{os.getpid()}"
    df.to_pickle(tmp)
    _atomic_replace(tmp, path)


def reject_flag_path(tiff_path: str) -> str:
    """Same convention as FileSelectionGUITiffOnly._get_reject_flag_path."""
    return os.path.splitext(tiff_path)[0] + "_rejected.flag"


def is_rejected(combined_df: pd.DataFrame, group: str, set_label: float, tiff_path: str) -> bool:
    """Flag file or pkl reject column (toggle_reject keeps both in step)."""
    if os.path.exists(reject_flag_path(tiff_path)):
        return True
    m = set_rows(combined_df, group, set_label)
    if "reject" in combined_df.columns and m.any():
        v = pd.to_numeric(combined_df.loc[m, "reject"], errors="coerce").fillna(0)
        return bool((v >= 1).any())
    return False


class ReviewSession:
    """State and file operations of the viewer (no Qt; tested headless)."""

    def __init__(self, pkl_path: str, csv_path: str | None = None, *, ch_1or2: int = 2, z_plus_minus: int = 1,
                 photon_threshold: int = 15, total_photon_threshold: int = 1000,
                 skip_lifetime_analysis: bool = True, uncaging_roi_keyframe_count: int | None = None,
                 show_alignment_r: bool = True, time_window_min: tuple[float, float] | None = None):
        self.pkl_path = pkl_path
        self.csv_path = csv_path or default_csv_for_pkl(pkl_path)
        self.ch, self.zpm = ch_1or2, z_plus_minus
        self.photon_threshold, self.total_photon_threshold = photon_threshold, total_photon_threshold
        self.skip_lifetime_analysis = skip_lifetime_analysis
        self.uncaging_roi_keyframe_count = uncaging_roi_keyframe_count
        self.show_alignment_r = show_alignment_r
        self.df = pd.read_pickle(pkl_path)
        if "after_align_full_save_path" in self.df.columns:
            from gui_roi_respan_seg_masks import set_uncaging_display_columns

            self.df["after_align_save_path"] = self.df["after_align_full_save_path"].fillna(
                self.df.get("after_align_save_path"))
            # ROI GUI marker = uncaging position on the uncaging frames (uncaging_xy_in_tiff)
            self.df = set_uncaging_display_columns(self.df)
        self.q = pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
        if time_window_min is not None:
            from gui_roi_respan_seg_masks import apply_time_window

            _, outdated = apply_time_window(self.df, time_window_min)
            if outdated:
                print(f"WARNING: this pkl was analyzed without the time window {time_window_min} min; "
                      "frames outside it are still shown. Re-run the analysis "
                      "(start_with_viewer = False) to remove them.")
        self.sets = list_sets(self.df)
        self.i = 0
        self._backed_up = False

    # -- navigation
    @property
    def current(self) -> tuple[str, float, str]:
        return self.sets[self.i]

    def move(self, step: int) -> None:
        if self.sets:
            self.i = int(np.clip(self.i + step, 0, len(self.sets) - 1))

    def rejected(self, k: int | None = None) -> bool:
        g, s, tiff = self.sets[self.i if k is None else k]
        return is_rejected(self.df, g, s, tiff)

    def has_quant(self) -> bool:
        g, s, _ = self.current
        return bool(set_rows(self.q, g, s, csv_like=True).any())

    def uncaging_xy(self) -> tuple[float, float] | None:
        """Uncaging position in GUI TIFF pixels (gui_roi_respan_seg_masks.uncaging_xy_in_tiff)."""
        from gui_roi_respan_seg_masks import uncaging_xy_in_tiff

        g, s, tiff = self.current
        return uncaging_xy_in_tiff(self.df[set_rows(self.df, g, s)], tiff)

    def review(self):
        g, s, tiff = self.current
        sq = self.q[set_rows(self.q, g, s, csv_like=True)] if len(self.q) else pd.DataFrame()
        notes = []
        if not len(sq):
            notes.append("not quantified")
            sq = pd.DataFrame(columns=["phase", "elapsed_time_sec", "file_path", "nAveFrame"])
        corr = None
        if self.show_alignment_r:
            try:
                from alignment_correlation import set_alignment_r

                sdf = self.df[set_rows(self.df, g, s)]
                gdf = self.df[(self.df["group"].astype(str) == str(g))
                              & (self.df["filepath_without_number"] == sdf["filepath_without_number"].iloc[0])]
                corr = set_alignment_r(sdf, gdf, tiff, ch=self.ch)
            except Exception as exc:  # correlation is informative only
                print("alignment correlation not available:", repr(exc))
        r = load_set_review(tiff, sq, title=f"[{self.i + 1}/{len(self.sets)}]  {g} set {s:g}", ch=self.ch,
                            corr=corr)
        r.notes = notes + r.notes
        r.rejected = self.rejected()
        r.uncaging_xy = self.uncaging_xy()
        return r

    # -- file updates
    def _backup_once(self) -> None:
        if self._backed_up or not os.path.exists(self.csv_path):
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(self.csv_path, f"{self.csv_path}.bak_{stamp}")
        self._backed_up = True

    def requantify_current(self) -> int:
        """Re-create raw masks and quantify only the current set; returns the new row count."""
        import gui_roi_fast_simple as g
        from gui_roi_respan_seg_masks import RESPAN_ROI_TYPES

        grp, s, _ = self.current
        sdf = self.df[set_rows(self.df, grp, s)].copy()
        sdf["reject"] = 0  # rejected sets are quantified too, so they can be looked at
        g.save_drift_corrected_roi_masks(sdf, roi_types=RESPAN_ROI_TYPES)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "one_set.csv")
            g.quantify_intensity_from_flim(
                sdf, self.ch, self.zpm, out, photon_threshold=self.photon_threshold,
                total_photon_threshold=self.total_photon_threshold,
                skip_lifetime_analysis=self.skip_lifetime_analysis,
                background_mode=g.BACKGROUND_MODE_MIP_P20,
            )
            new_q = pd.read_csv(out) if os.path.exists(out) else None
        self._backup_once()
        self.q = replace_set_rows(self.q, new_q, grp, s)
        atomic_write_csv(self.q, self.csv_path)
        return 0 if new_q is None else len(new_q)

    def _update_pkl(self, values: dict) -> None:
        g, s, _ = self.current
        m = set_rows(self.df, g, s)
        for k, v in values.items():
            self.df.loc[m, k] = v
        atomic_write_pickle(self.df, self.pkl_path)

    def toggle_reject(self) -> bool:
        """Flip reject for the current set (flag file + pkl); data and display are kept."""
        _, _, tiff = self.current
        new_state = not self.rejected()
        flag = reject_flag_path(tiff)
        if new_state:
            with open(flag, "w", encoding="utf-8") as fh:
                fh.write(f"Rejected at: {datetime.datetime.now().isoformat()} (set_review_viewer)\n")
        elif os.path.exists(flag):
            os.remove(flag)
        self._update_pkl({"reject": 1 if new_state else 0})
        return new_state



# --------------------------------------------------------------------------- Qt viewer
def _edit_rois_in_gui(session: ReviewSession) -> bool:
    """Open the existing ROI GUI for Spine, then Shaft. True if any mask file changed."""
    from gui_integration_tiff_only import launch_roi_analysis_gui_tiff_only
    from gui_roi_respan_seg_masks import RESPAN_ROI_TYPES

    grp, s, tiff = session.current
    base = os.path.splitext(tiff)[0]

    def stamp(p):
        return os.path.getmtime(p) if os.path.exists(p) else None

    before = {t: stamp(f"{base}_{t}_roi_mask.tif") for t in RESPAN_ROI_TYPES}
    for roi_type in RESPAN_ROI_TYPES:
        res = launch_roi_analysis_gui_tiff_only(
            session.df, tiff, grp, s, header=roi_type, save_tiff_path=f"{base}_{roi_type}_roi_mask.tif",
            shortcut_host=None, uncaging_roi_keyframe_count=session.uncaging_roi_keyframe_count,
        )
        if not isinstance(res, dict) or res.get("exit_kind") == "cancel":
            break
    return any(stamp(f"{base}_{t}_roi_mask.tif") != before[t] for t in RESPAN_ROI_TYPES)


def make_viewer(session: ReviewSession, edit_fn=_edit_rois_in_gui):
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

    class SetReviewViewer(QMainWindow):
        def __init__(self):
            super().__init__()
            self.session = session
            self.crop_axes = []
            self.setWindowTitle("Set review  (Left/Right: move, E: edit ROI, R: reject, Q: close)")
            self.resize(1700, 760)
            w = QWidget()
            self.setCentralWidget(w)
            lay = QVBoxLayout(w)
            bar = QHBoxLayout()
            self.combo = QComboBox()
            self.combo.setFocusPolicy(Qt.ClickFocus)
            self.combo.activated.connect(self._jump)
            bar.addWidget(self.combo, 3)
            self.reject_button = None
            for text, fn in (("< Prev", lambda: self._move(-1)), ("Next >", lambda: self._move(1)),
                             ("Edit ROI (E)", self.edit), ("Reject (R)", self.reject)):
                b = QPushButton(text)
                b.setFocusPolicy(Qt.NoFocus)
                b.clicked.connect(fn)
                bar.addWidget(b)
                if text.startswith("Reject"):
                    self.reject_button = b
            self.status = QLabel("")
            bar.addWidget(self.status, 4)
            lay.addLayout(bar)
            self.fig = Figure(figsize=(16, 6))
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setFocusPolicy(Qt.NoFocus)
            lay.addWidget(self.canvas)
            self.setFocusPolicy(Qt.StrongFocus)
            self._fill_combo()
            self.redraw()

        def _fill_combo(self):
            self.combo.blockSignals(True)
            self.combo.clear()
            for k, (g, s, _) in enumerate(self.session.sets):
                mark = "[REJECTED] " if self.session.rejected(k) else ""
                self.combo.addItem(f"{k + 1}: {mark}{g} set {s:g}")
            self.combo.blockSignals(False)

        def _update_combo_item(self):
            k = self.session.i
            g, s, _ = self.session.sets[k]
            mark = "[REJECTED] " if self.session.rejected(k) else ""
            self.combo.setItemText(k, f"{k + 1}: {mark}{g} set {s:g}")

        def redraw(self, message: str = ""):
            if not self.session.sets:
                self.status.setText("no sets with a full-size TIFF")
                return
            if not self.session.has_quant():
                self.status.setText("quantifying this set ...")
                self.repaint()
                try:
                    n = self.session.requantify_current()
                    message = message or f"quantified ({n} rows)"
                except Exception as exc:
                    message = f"quantification failed: {exc!r}"
            try:
                self.crop_axes = draw_set_review(self.fig, self.session.review())
            except Exception as exc:  # show the problem, keep the viewer usable
                self.fig.clf()
                self.crop_axes = []
                self.fig.text(0.02, 0.5, f"cannot draw this set: {exc!r}", color="red")
            self.canvas.draw_idle()
            self.combo.setCurrentIndex(self.session.i)
            if self.reject_button is not None:
                self.reject_button.setText("Un-reject (R)" if self.session.rejected() else "Reject (R)")
            self.status.setText(message)
            self.setFocus()

        def _move(self, step):
            self.session.move(step)
            self.redraw()

        def _jump(self, idx):
            self.session.i = int(idx)
            self.redraw()

        def edit(self):
            changed = edit_fn(self.session)
            if changed:
                self.status.setText("re-quantifying ...")
                self.repaint()
                n = self.session.requantify_current()
                self.redraw(f"ROI updated; re-quantified ({n} rows)")
            else:
                self.redraw("no ROI change")

        def reject(self):
            state = self.session.toggle_reject()
            self._update_combo_item()
            self.redraw("rejected (R again to undo)" if state else "un-rejected")

        def keyPressEvent(self, ev):
            k = ev.key()
            if k == Qt.Key_Right:
                self._move(1)
            elif k == Qt.Key_Left:
                self._move(-1)
            elif k in (Qt.Key_E, Qt.Key_Return, Qt.Key_Enter):
                self.edit()
            elif k == Qt.Key_R:
                self.reject()
            elif k in (Qt.Key_Q, Qt.Key_Escape):
                self.close()
            else:
                super().keyPressEvent(ev)

    return SetReviewViewer()


def launch_set_review_viewer(pkl_path: str | None = None, csv_path: str | None = None, **session_kwargs):
    """Open the viewer and block until it is closed. Without pkl_path a file dialog asks for it."""
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    if not pkl_path:
        from simple_dialog import ask_open_path_gui

        pkl_path = ask_open_path_gui(filetypes=[("combined_df pickle", "*.pkl")])
        if not pkl_path:
            print("No pkl selected.")
            return None
    session = ReviewSession(pkl_path, csv_path, **session_kwargs)
    print(f"Set review: {len(session.sets)} sets, CSV {session.csv_path}")
    viewer = make_viewer(session)
    viewer.show()
    app.exec_()
    return session


def main() -> None:
    ap = argparse.ArgumentParser(description="Set-by-set review viewer.")
    ap.add_argument("--pkl", default=None, help="combined_df pkl (dialog if omitted)")
    ap.add_argument("--csv", default=None, help="all-frames CSV (default: <pkl>" + CSV_SUFFIX + ")")
    ap.add_argument("--ch", type=int, default=2)
    ap.add_argument("--z-plus-minus", type=int, default=1)
    ap.add_argument("--with-lifetime", action="store_true")
    ap.add_argument("--uncaging-roi-keyframe-count", type=int, default=3)
    a = ap.parse_args()
    launch_set_review_viewer(a.pkl, a.csv, ch_1or2=a.ch, z_plus_minus=a.z_plus_minus,
                             skip_lifetime_analysis=not a.with_lifetime,
                             uncaging_roi_keyframe_count=a.uncaging_roi_keyframe_count)


if __name__ == "__main__":
    main()
