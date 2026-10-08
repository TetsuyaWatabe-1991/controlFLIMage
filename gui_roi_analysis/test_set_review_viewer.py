"""Headless tests for set_review_viewer.py (offscreen Qt).

Unit tests use synthetic tables. The integration test needs prepared set folders
(combined_df_one_set.pkl, quant.csv, tif/ with TIFF + Type-A masks + frame_info), e.g.
made by rebuilding sets of a session into a scratch folder. They are copied to a
temporary folder first; the FLIM files they point to are only read.

Run:
    python test_set_review_viewer.py
    python test_set_review_viewer.py --sets <folder1> <folder2>
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(HERE))
import set_review_viewer as v  # noqa: E402


def test_replace_set_rows_keeps_order():
    q = pd.DataFrame({"group": list("aabbcc"), "set_label": [0, 0, 1, 1, 0, 0], "x": range(6)})
    new = pd.DataFrame({"group": ["b"] * 3, "set_label": [1] * 3, "x": [10, 11, 12]})
    r = v.replace_set_rows(q, new, "b", 1)
    assert r.x.tolist() == [0, 1, 10, 11, 12, 4, 5]
    r2 = v.replace_set_rows(q, None, "b", 1)
    assert r2.x.tolist() == [0, 1, 4, 5]
    r3 = v.replace_set_rows(q, new.assign(group="d"), "d", 1)
    assert r3.x.tolist()[-3:] == [10, 11, 12]


def _prepare(folders: list[str], td: str) -> tuple[str, str]:
    dfs, qs = [], []
    for k, src in enumerate(folders):
        dst = os.path.join(td, f"set{k}")
        shutil.copytree(src, dst)
        df = pd.read_pickle(os.path.join(dst, "combined_df_one_set.pkl"))
        for col in ("after_align_save_path", "after_align_full_save_path"):
            df[col] = df[col].map(lambda p: os.path.join(dst, "tif", os.path.basename(str(p))) if isinstance(p, str) else p)
        dfs.append(df)
        qs.append(pd.read_csv(os.path.join(dst, "quant.csv")))
    pkl, csv = os.path.join(td, "combined.pkl"), os.path.join(td, "all_frames.csv")
    pd.concat(dfs, ignore_index=True).to_pickle(pkl)
    pd.concat(qs, ignore_index=True).to_csv(csv, index=False)
    return pkl, csv


def integration(folders: list[str]) -> None:
    import tifffile
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    import flim_fast_io as ffi

    ffi._save_cached_intensity = lambda *a, **k: None  # never write into the session folder
    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as td:
        pkl, csv = _prepare(folders, td)
        q0 = pd.read_csv(csv)
        edited = []

        def fake_edit(session):  # stands in for the ROI GUI: move the Spine ROI 2 px right
            base = os.path.splitext(session.current[2])[0]
            p = f"{base}_Spine_roi_mask.tif"
            m = tifffile.imread(p)
            tifffile.imwrite(p, np.roll(m, 2, axis=2).astype(np.uint8))
            edited.append(session.current[:2])
            return True

        s = v.ReviewSession(pkl, csv, ch_1or2=2, z_plus_minus=1)
        w = v.make_viewer(s, edit_fn=fake_edit)
        w.show()
        app.processEvents()
        assert len(s.sets) == len(folders) and s.i == 0
        QTest.keyClick(w, Qt.Key_Right)
        assert s.i == 1, s.i
        QTest.keyClick(w, Qt.Key_Right)
        assert s.i == 1, "stays at the last set"
        QTest.keyClick(w, Qt.Key_Left)
        assert s.i == 0
        g, sl, tiff = s.current
        other = [x for x in s.sets if x[:2] != (g, sl)][0]

        # edit -> only this set re-quantified, other set untouched
        QTest.keyClick(w, Qt.Key_E)
        q1 = pd.read_csv(csv)
        m0, m1 = v.set_rows(q0, g, sl, csv_like=True), v.set_rows(q1, g, sl, csv_like=True)
        o0, o1 = v.set_rows(q0, *other[:2], csv_like=True), v.set_rows(q1, *other[:2], csv_like=True)
        assert edited == [(g, sl)] and m1.sum() == m0.sum()
        assert not np.allclose(q0[m0].Spine_Ch2_intensity.to_numpy(), q1[m1].Spine_Ch2_intensity.to_numpy())
        assert q0[o0].reset_index(drop=True).equals(q1[o1].reset_index(drop=True))
        assert any(f.startswith(os.path.basename(csv) + ".bak_") for f in os.listdir(td)), "backup written"
        print(f"  edit: {g} set {sl:g} re-quantified ({m1.sum()} rows), other set unchanged")

        # reject -> flag + pkl; the set stays in the list with its data; R again restores
        n_sets = len(s.sets)
        QTest.keyClick(w, Qt.Key_R)
        assert os.path.exists(v.reject_flag_path(tiff))
        assert (pd.read_pickle(pkl).pipe(lambda d: d[v.set_rows(d, g, sl)]).reject == 1).all()
        q2 = pd.read_csv(csv)
        assert v.set_rows(q2, g, sl, csv_like=True).sum() == m1.sum(), "rows kept after reject"
        assert len(s.sets) == n_sets and s.review().rejected and "[REJECTED]" in w.combo.itemText(s.i)
        assert w.reject_button.text().startswith("Un-reject")
        QTest.keyClick(w, Qt.Key_R)
        assert not os.path.exists(v.reject_flag_path(tiff))
        assert (pd.read_pickle(pkl).pipe(lambda d: d[v.set_rows(d, g, sl)]).reject == 0).all()
        assert not s.review().rejected and "[REJECTED]" not in w.combo.itemText(s.i)
        assert pd.read_csv(csv).equals(q2)
        print("  reject / un-reject: flag and pkl updated, set and data kept visible")

        # uncaging position: yellow cross at the FLIM-header position on the uncaging crops only
        import gui_roi_respan_seg_masks as rs

        import flim_fast_io as ffi
        from FLIMageFileReader2 import FileReader

        xy = s.uncaging_xy()
        sdf = s.df[v.set_rows(s.df, g, sl)]
        unc_path = sdf[sdf.phase == "unc"].file_path.iloc[0]
        rd = FileReader()
        rd.read_imageFile(unc_path, False)
        pos = rd.statedict["State.Uncaging.Position"]  # what FLIMage draws (0-1 of the field)
        hx = pos[0] * rd.statedict["State.Acq.pixelsPerLine"]
        hy = pos[1] * rd.statedict["State.Acq.linesPerFrame"]
        # the tissue around the cross on the GUI TIFF uncaging frame = the tissue around
        # FLIMage's cross on the raw uncaging image (same pixels, not corrected towards the spine)
        fi = pd.read_csv(os.path.splitext(tiff)[0] + "_frame_info.csv")
        k_unc = int(fi.index[fi.phase.astype(str).str.startswith("unc")][0])
        tiff_unc0 = tifffile.imread(tiff)[k_unc]
        raw_unc0 = ffi.uncaging_frames_from_intensity(ffi.load_flim_intensity(unc_path, use_cache=True)[0], 2)[0]
        r = 4
        rnd = lambda val: int(np.floor(val + 0.5))  # same rounding for both (no banker's rounding)
        tx, ty, rx, ry = rnd(xy[0]), rnd(xy[1]), rnd(hx), rnd(hy)
        patch_tiff = tiff_unc0[ty - r:ty + r + 1, tx - r:tx + r + 1]
        patch_raw = raw_unc0[ry - r:ry + r + 1, rx - r:rx + r + 1]
        assert patch_tiff.shape == patch_raw.shape and np.array_equal(patch_tiff, patch_raw), "cross not on FLIMage tissue"
        y0, _, x0, _ = s.review().crop_box
        for ax, f in w.crop_axes:
            crosses = [ln for ln in ax.get_lines() if ln.get_marker() == "+"]
            if f.phase.startswith("unc"):
                assert len(crosses) == 1 and crosses[0].get_color() == "yellow"
                cx, cy = crosses[0].get_xdata()[0], crosses[0].get_ydata()[0]
                assert np.isclose(cx + x0, xy[0]) and np.isclose(cy + y0, xy[1])
            else:
                assert not crosses
        # the ROI GUI marker (uncaging_display_x/y) is the same point: in the viewer's df
        # (used when E opens the ROI GUI) and after the workflow's _prepare step
        for d in (s.df, rs._prepare_combined_df_for_roi_gui(pd.read_pickle(pkl))):
            dm = d[v.set_rows(d, g, sl)]
            assert np.allclose(dm.uncaging_display_x, xy[0]) and np.allclose(dm.uncaging_display_y, xy[1])
        print(f"  uncaging position: yellow cross at x={xy[0]:.1f}, y={xy[1]:.1f} on the uncaging frames "
              f"(= ROI GUI marker)")
        w.close()
        app.processEvents()

        # start from the viewer with only the pkl: default CSV name, missing sets quantified on display
        default_csv = v.default_csv_for_pkl(pkl)
        qall = pd.read_csv(csv)
        g2, sl2 = other[:2]
        v.atomic_write_csv(qall[~v.set_rows(qall, g2, sl2, csv_like=True)], default_csv)
        s2 = v.ReviewSession(pkl, None)
        assert s2.csv_path == default_csv
        w2 = v.make_viewer(s2, edit_fn=fake_edit)
        w2.show()
        app.processEvents()
        s2.i = [k for k, x in enumerate(s2.sets) if x[:2] == (g2, sl2)][0]
        w2.redraw()
        q4 = pd.read_csv(default_csv)
        assert v.set_rows(q4, g2, sl2, csv_like=True).sum() == v.set_rows(qall, g2, sl2, csv_like=True).sum()
        print("  start with viewer: pkl only, missing set quantified when shown")
        w2.close()
        app.processEvents()
    print("integration passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=[])
    a = ap.parse_args()
    test_replace_set_rows_keeps_order()
    print("PASS test_replace_set_rows_keeps_order")
    if a.sets:
        integration(a.sets)
