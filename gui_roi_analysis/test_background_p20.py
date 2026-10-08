"""Headless tests: background = 20th percentile of the quantified image; GUI without BG ROI.

Run:
    python test_background_p20.py
    python test_background_p20.py --real G:\\ImagingData\\Tetsuya\\20260925\\auto1\\combined_df_respan.pkl
The --real check quantifies two sets into a temporary folder and compares the
Background column with np.percentile(MIP, 20) computed independently from the FLIM
files (session files are only read).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import gui_roi_fast_simple as g  # noqa: E402


class _Acq:
    nAveFrame = 1


class _State:
    Acq = _Acq()


class _Info:
    State = _State()


def _row(imagearray, roi_raw, mode, phase="pre", zw=(1, 4)):
    return g._row_from_flim_data(
        imagearray, _Info(), "x.flim", 0, 0, phase, 0.0, "", roi_raw, 2, 1, 0, "g", None, 80e6, 15, 1000,
        skip_lifetime_analysis=True, z_window_override=zw, background_mode=mode,
    )


def _synthetic():
    rng = np.random.default_rng(0)
    arr = rng.poisson(0.3, size=(6, 1, 2, 32, 32, 4)).astype(np.uint16)  # Z, T, C, Y, X, bins
    arr[2, 0, 1, 10:14, 10:14, :] += 20  # a bright spine in z=2
    spine = np.zeros((1, 32, 32), np.uint8)
    spine[0, 10:14, 10:14] = 1
    bg_on_tissue = np.zeros((1, 32, 32), np.uint8)
    bg_on_tissue[0, 9:15, 9:15] = 1  # a badly placed Background ROI
    return arr, spine, bg_on_tissue


def test_p20_of_quantified_mip():
    arr, spine, bad_bg = _synthetic()
    row = _row(arr, {"Spine": spine, "Background": bad_bg}, g.BACKGROUND_MODE_MIP_P20)
    mip = arr[1:4, 0, 1].sum(-1).max(0).astype(np.float32)
    assert row["Background_Ch2_intensity"] == float(np.percentile(mip, 20))
    assert row["Spine_Ch2_intensity"] == float(mip[spine[0] > 0].mean())
    # the Background ROI is ignored in p20 mode
    legacy = _row(arr, {"Spine": spine, "Background": bad_bg}, g.BACKGROUND_MODE_ROI)
    assert legacy["Background_Ch2_intensity"] > 10 * max(row["Background_Ch2_intensity"], 0.1)


def test_p20_without_background_roi_and_uncaging_frame():
    arr, spine, _ = _synthetic()
    row = _row(arr, {"Spine": spine}, g.BACKGROUND_MODE_MIP_P20)
    assert "Background_Ch2_intensity" in row and "Background_Ch1_intensity" in row
    legacy = _row(arr, {"Spine": spine}, g.BACKGROUND_MODE_ROI)
    assert "Background_Ch2_intensity" not in legacy
    unc = _row(arr, {"Spine": spine}, g.BACKGROUND_MODE_MIP_P20, phase="unc", zw=None)
    frame = arr[0, 0, 1].sum(-1).astype(np.float32)
    assert unc["Background_Ch2_intensity"] == float(np.percentile(frame, 20))


def test_gui_table_without_bg():
    from PyQt5.QtWidgets import QApplication, QPushButton

    from file_selection_gui_tiff_only import FileSelectionGUITiffOnly
    from gui_roi_respan_seg_masks import RESPAN_ROI_TYPES

    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as td:
        tiff = os.path.join(td, "g_0_after_align_full.tif")
        import tifffile

        tifffile.imwrite(tiff, np.zeros((3, 8, 8), np.float32))
        df = pd.DataFrame({"filepath_without_number": ["G:/x/p__highmag_1_"] * 3, "group": ["g"] * 3,
                           "nth_set_label": [0, 0, 0], "phase": ["pre", "unc", "post"], "nth_omit_induction": [0, -1, 1],
                           "file_path": ["a.flim", "b.flim", "c.flim"], "after_align_save_path": [tiff] * 3})
        for roi_types, n_btn, has_bg in ((RESPAN_ROI_TYPES, 2, False), (None, 3, True)):
            gui = FileSelectionGUITiffOnly(df.copy(), None, [], False, roi_types=roi_types)
            headers = [gui.table.horizontalHeaderItem(i).text() for i in range(gui.table.columnCount())]
            assert ("BG" in headers) == has_bg, headers
            last = gui.table.cellWidget(0, gui.table.columnCount() - 1)
            buttons = [b.text() for b in last.findChildren(QPushButton)]
            assert len(buttons) == n_btn, buttons
            gui.close()
    app.processEvents()


def real_check(pkl: str) -> None:
    import tifffile

    import flim_fast_io as ffi
    from FLIMageFileReader2 import FileReader

    ffi._save_cached_intensity = lambda *a, **k: None
    df_all = pd.read_pickle(pkl)
    keys = ["filepath_without_number", "group", "nth_set_label"]
    sets = [k for k, d in df_all[df_all.nth_set_label != -1].groupby(keys, sort=False) if (d.phase == "unc").any()][:2]
    with tempfile.TemporaryDirectory() as td:
        cdf = df_all.copy()
        for k in sets:
            sdf = cdf[(cdf[keys[0]] == k[0]) & (cdf.group == k[1]) & (cdf.nth_set_label == k[2])]
            fl = sdf[sdf.phase.isin(["pre", "post"])].file_path.tolist()
            shp = ffi.load_flim_intensity(fl[0], use_cache=True)[0].shape
            g._rebuild_one_set_full_size(
                combined_df=cdf, each_set_df=sdf, each_set_label=k[2], each_group=k[1], ch=2, z_plus_minus=1,
                Z_full=int(shp[0]), Y_full=int(shp[-2]), X_full=int(shp[-1]), Aligned_4d_array=np.empty((0, 1, 1, 1)),
                file_path_to_array_idx={}, n_aligned=0,
                runtime_shift_map={r.file_path: (float(r.shift_y), float(r.shift_x)) for r in sdf.itertuples()
                                   if r.phase in ("pre", "post")},
                tif_savefolder=td, skip_tiff_if_exists=False, error_log=[], fast_mode=True, intensity_cache={},
                runtime_shift_z_map={r.file_path: float(r.shift_z) for r in sdf.itertuples()
                                     if r.phase in ("pre", "post")},
            )
        sub = cdf.set_index(keys).loc[sets].reset_index()
        sub["after_align_save_path"] = sub["after_align_full_save_path"]
        sub["reject"] = 0
        for k in sets:
            tiff = sub[(sub.group == k[1]) & (sub.nth_set_label == k[2])].after_align_save_path.iloc[0]
            st = tifffile.imread(tiff)
            m = np.zeros(st.shape, np.uint8)
            m[:, 60:68, 60:68] = 1
            for t in ("Spine", "DendriticShaft"):
                tifffile.imwrite(os.path.splitext(tiff)[0] + f"_{t}_roi_mask.tif", m)
        g.save_drift_corrected_roi_masks(sub, roi_types=["Spine", "DendriticShaft"])
        out = os.path.join(td, "q.csv")
        g.quantify_intensity_from_flim(sub, 2, 1, out, skip_lifetime_analysis=True,
                                       background_mode=g.BACKGROUND_MODE_MIP_P20)
        q = pd.read_csv(out)
        n = 0
        for r in q[q.phase.isin(["pre", "post"])].itertuples():
            rd = FileReader()
            rd.read_imageFile(r.file_path, True)
            mip = np.array(rd.image)[int(r.z_from):int(r.z_to), 0, 1].sum(-1).max(0)
            assert float(r.Background_Ch2_intensity) == float(np.percentile(mip, 20)), r.file_path
            n += 1
        vals = q.Background_Ch2_intensity.value_counts().sort_index().to_dict()
        print(f"real: {n} pre/post frames match np.percentile(MIP, 20); Background values (all frames): {vals}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    a = ap.parse_args()
    k = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
            k += 1
    print(f"all {k} tests passed")
    if a.real:
        real_check(a.real)
