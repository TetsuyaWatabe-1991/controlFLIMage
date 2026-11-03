"""Headless tests for respan_tracked_spine_roi.py (synthetic data, no RESPAN / GUI).

Run: python test_respan_tracked_spine_roi.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import respan_tracked_spine_roi as m

XY_UM = 0.1


def disk(shape, cy, cx, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.uint8)


def test_int_shift_matches_ndimage():
    rng = np.random.default_rng(0)
    a = (rng.random((20, 24)) > 0.7).astype(np.uint8)
    for dy, dx in [(0, 0), (3, -2), (-5, 4), (-19, 23), (30, 0)]:
        ref = ndi.shift(a, (dy, dx), order=0, mode="constant", cval=0)
        assert np.array_equal(m._int_shift(a, dy, dx), ref), (dy, dx)


def _frames(shifts):
    return [{"flim": f"C:/d/f_{i:03d}.flim", "crop_stem": f"f_{i:03d}__track_ch2_zyx", "shift_zyx": list(s)}
            for i, s in enumerate(shifts)]


def test_tracking_follows_spine_and_carries_lost_frame():
    # Stage drift: raw position = true(002 frame) - shift. Spine also grows 0.3 um/frame in x.
    shifts = [(0, 0, 0), (0, 2, -1), (1, 4, -3), (1, 5, -4), (0, 6, -6)]
    frames = _frames(shifts)
    true_y, true_x, true_z = 50.0, 60.0, 10.0
    rows = {}
    for i, (f, (sz, sy, sx)) in enumerate(zip(frames, shifts)):
        y, x, z = true_y - sy, true_x + 3 * i - sx, true_z - sz
        spine = {"y": y, "x": x, "z": z, "spine_id": 1}
        distractor = {"y": y + 12, "x": x + 12, "z": z, "spine_id": 2}  # 1.7 um away
        rows[f["crop_stem"]] = [] if i == 3 else [distractor, spine]  # frame 3: RESPAN missed it
    last_pre = frames[2]["flim"]
    unc = (true_y - shifts[2][1] + 3, true_x + 6 - shifts[2][2] - 4)  # uncaging point 0.5 um from head
    picks = m.track_frames(frames, rows, unc, last_pre, XY_UM)
    assert [p["status"] for p in picks] == ["tracked", "tracked", "tracked", "lost", "tracked"]
    assert all(p["row"] is None or p["row"]["spine_id"] == 1 for p in picks)
    # Lost frame: previous head carried by the shift difference.
    h2, h3 = picks[2]["head"], picks[3]["head"]
    assert np.allclose(h3, [h2[0] + shifts[2][0] - shifts[3][0], h2[1] + shifts[2][1] - shifts[3][1],
                            h2[2] + shifts[2][2] - shifts[3][2]])
    # Step > 1 um is refused (distractor only).
    rows2 = {k: [r for r in v if r["spine_id"] == 2] for k, v in rows.items()}
    rows2[frames[0]["crop_stem"]] = rows[frames[0]["crop_stem"]]
    p2 = m.track_frames(frames, rows2, unc, last_pre, XY_UM)
    assert [p["status"] for p in p2][1:] == ["lost"] * 4


def test_type_a_stack_covers_spine_in_gui_frames():
    """Synthetic raw frames -> GUI frames (integer shift vs first pre) -> ROI must cover the spine."""
    shape = (80, 90)
    shifts = [(0, 0, 0), (0, 2.4, -1.2), (0, 4.6, -3.1), (0, 5.2, -4.4), (0, 6.0, -6.5)]
    frames = _frames(shifts)
    heads = [(40 - s[1] + d, 45 - s[2] + d) for s, d in zip(shifts, [0, 0, 1, 1, 2])]  # raw heads
    picks = [{"flim": f["flim"], "crop_stem": f["crop_stem"], "shift_zyx": f["shift_zyx"],
              "status": "tracked", "row": {}, "step_um": 0.0, "head": [10, hy, hx]}
             for f, (hy, hx) in zip(frames, heads)]
    ref = picks[2]  # last pre
    ref_roi = disk(shape, round(ref["head"][1]), round(ref["head"][2]), 4)
    phases = ["pre", "pre", "pre", "uncaging", "uncaging", "post", "post"]
    names = ["f_000.flim", "f_001.flim", "f_002.flim", "unc.flim", "unc.flim", "f_003.flim", "f_004.flim"]
    src = [0, 1, 2, None, None, 3, 4]
    s0 = shifts[0]
    fi = pd.DataFrame({"phase": phases, "filename": names,
                       "shift_y": [0 if k is None else int(round(shifts[k][1] - s0[1])) for k in src],
                       "shift_x": [0 if k is None else int(round(shifts[k][2] - s0[2])) for k in src]})
    stack, notes = m.build_type_a_stack(picks, ref, ref_roi, fi)
    assert stack.shape == (7,) + shape
    assert notes[3] == notes[4] == "uncaging=last pre"
    assert np.array_equal(stack[3], stack[2]) and np.array_equal(stack[4], stack[2])
    for i, k in enumerate(src):
        if k is None:
            continue
        raw = disk(shape, round(heads[k][0]), round(heads[k][1]), 2)  # spine image in raw frame k
        gui = ndi.shift(raw, (fi.shift_y[i], fi.shift_x[i]), order=0)  # as _rebuild_one_set_full_size
        assert gui.sum() > 0 and np.all(stack[i][gui > 0] > 0), f"frame {i}: ROI misses the spine"
        assert stack[i].sum() == ref_roi.sum(), "ROI shape must be fixed"


def test_replaceable_rules():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "g_0_after_align_full_Spine_roi_mask.tif")
        seg = disk((30, 30), 15, 15, 3)
        assert m._is_replaceable(p, seg)[0]
        tifffile.imwrite(p, np.stack([seg] * 4))
        assert m._is_replaceable(p, seg)[0]  # seg init
        own = np.stack([disk((30, 30), 12, 12, 3)] * 4)
        tifffile.imwrite(p, own)
        side = os.path.join(td, "g_0_after_align_full_Spine_roi_mask_source.json")
        with open(side, "w") as fh:
            json.dump({"sha1": m._sha(own > 0)}, fh)
        assert m._is_replaceable(p, seg)[0]  # own, unedited
        edited = own.copy()
        edited[1, 0, 0] = 1
        tifffile.imwrite(p, edited)
        assert not m._is_replaceable(p, seg)[0]  # edited after tracking
        os.remove(side)
        assert not m._is_replaceable(p, seg)[0]  # manual mask, no sidecar
        tifffile.imwrite(p, np.stack([seg] * 4))
        with open(side, "w") as fh:
            json.dump({"sha1": "stale"}, fh)
        assert m._is_replaceable(p, seg)[0]  # re-seeded from seg (overwrite_seg_roi_masks)


def test_subtract_shaft_removes_overlap_and_keeps_empty_frames():
    spine = np.stack([disk((30, 30), 15, 15, 4)] * 3).astype(np.uint8)
    shaft = np.zeros_like(spine)
    shaft[0, :, 17:] = 1   # overlaps the right part of the spine
    shaft[2] = 1           # would remove the whole ROI -> frame kept
    out, removed = m.subtract_shaft(spine, shaft)
    assert removed[0] > 0 and not (out[0] & shaft[0]).any() and out[0].any()
    assert removed[1] == 0 and np.array_equal(out[1], spine[1])
    assert removed[2] == 0 and np.array_equal(out[2], spine[2])
    assert out.dtype == spine.dtype


def test_trimmed_reference_keeps_roi_size_constant():
    # spine ROI next to a fixed shaft; later frames move the spine 3 px toward the shaft
    ref = np.zeros((40, 40), np.uint8)
    ref[10:16, 20:26] = 1                       # 36 px
    shaft = np.zeros((40, 40), np.uint8)
    shaft[16:22, :] = 1                         # horizontal shaft just below
    shift_y, shift_x = 2, -1                    # raw -> GUI shift of the reference row
    trimmed, removed = m.trim_reference_roi(ref, shaft, shift_y, shift_x)
    gui = m._int_shift(ref, shift_y, shift_x) > 0
    assert removed == int((gui & (shaft > 0)).sum()) > 0
    assert not (m._int_shift(trimmed, shift_y, shift_x).astype(bool) & (shaft > 0)).any()
    picks = [dict(flim="a_002.flim", head=(5, 13, 23), status="tracked"),
             dict(flim="a_003.flim", head=(5, 16, 23), status="tracked")]   # moved 3 px down
    fi = pd.DataFrame(dict(filename=["a_002.flim", "a_003.flim"], phase=["pre", "post"],
                           shift_y=[shift_y, shift_y], shift_x=[shift_x, shift_x]))
    stack, _ = m.build_type_a_stack(picks, picks[0], trimmed, fi)
    sizes = stack.reshape(len(stack), -1).sum(1)
    assert sizes[0] == sizes[1] == int(trimmed.sum()), sizes
    # the former frame-by-frame subtraction would have shrunk the moved frame
    old, _ = m.subtract_shaft(m.build_type_a_stack(picks, picks[0], ref, fi)[0],
                              np.stack([shaft, shaft]))
    old_sizes = old.reshape(len(old), -1).sum(1)
    assert old_sizes[1] < old_sizes[0], old_sizes
    # nothing left after trimming -> ROI kept
    full = np.ones((40, 40), np.uint8)
    kept, n = m.trim_reference_roi(ref, full, 0, 0)
    assert n == 0 and np.array_equal(kept, ref)


def test_apply_end_to_end_synthetic():
    """Queue job -> tracked Spine mask on disk; second run keeps a GUI edit untouched."""
    import csv
    import sys
    import types
    from dataclasses import dataclass

    from respan_online_queue import Job, JobQueue
    from respan_track_session import QUEUE_NAME, track_key

    shape = (60, 70)
    with tempfile.TemporaryDirectory() as td:
        hdir = os.path.join(td, "p1__highmag_1")
        tif_dir = os.path.join(td, "tif")
        os.makedirs(hdir)
        os.makedirs(tif_dir)
        flims = [os.path.join(td, f"p1__highmag_1_{i:03d}.flim") for i in (3, 4, 5, 7, 8)]
        shifts = [(0, 0, 0), (0, 1, 1), (0, 2, 3), (0, 3, 4), (0, 4, 4)]
        heads = [(5, 30 - s[1], 35 - s[2]) for s in shifts]  # raw heads of a still spine
        q = JobQueue(os.path.join(td, QUEUE_NAME))
        key = track_key(flims[2])
        jd = q.job_dir(key)
        os.makedirs(jd / "input")
        os.makedirs(jd / "run" / "Tables")
        crops, frames = {}, []
        for f, s, h in zip(flims, shifts, heads):
            st = os.path.basename(f)[:-5] + "__track_ch2_zyx"
            crops[st] = {"flim": f, "y0": 0, "x0": 0, "full_shape_yx": list(shape)}
            frames.append({"flim": f, "crop_stem": st, "shift_zyx": list(s)})
            with open(jd / "run" / "Tables" / f"{st}_detected_spines.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["spine_id", "x", "y", "z"])
                w.writerow([1, h[2], h[1], h[0]])
        (jd / "input" / "crops.json").write_text(json.dumps(crops))
        (jd / "input" / "track_info.json").write_text(json.dumps(
            {"spine_stem": "p1__highmag_1_id001", "uncaging_x_pix": heads[2][2] + 3,
             "uncaging_y_pix": heads[2][1], "last_pre_flim": flims[2], "xy_um": XY_UM, "frames": frames}))
        q.submit(Job(key=key, flim_path=flims[2], priority=1, deadline=None, source="test"))
        q.finish(q.take_next(), result_dir=str(jd / "run"))

        base = "g_0_after_align_full"
        tiff = os.path.join(tif_dir, base + ".tif")
        tifffile.imwrite(tiff, np.zeros((7,) + shape, np.float32))
        names = [os.path.basename(f) for f in flims]
        fi = pd.DataFrame({"phase": ["pre"] * 3 + ["uncaging"] * 2 + ["post"] * 2,
                           "filename": names[:3] + ["unc.flim"] * 2 + names[3:],
                           "shift_y": [0, 1, 2, 0, 0, 3, 4], "shift_x": [0, 1, 3, 0, 0, 4, 4]})
        fi.to_csv(os.path.join(tif_dir, base + "_frame_info.csv"), index=False)
        seg = disk(shape, 10, 10, 3)
        tifffile.imwrite(os.path.join(tif_dir, base + "_Spine_roi_mask.tif"), np.stack([seg] * 7))

        @dataclass
        class Rec:
            flim_path: str
            spine_stem: str = "p1__highmag_1_id001"

        fake = types.ModuleType("gui_roi_respan_seg_masks")
        fake.highmag_savefolder_from_filepath_without_number = lambda fp: hdir
        fake.match_uncaging_record_for_set = lambda sdf, recs: recs[0]
        fake.seg_mask_paths = lambda h, s: {"Spine": "seg"}
        fake._load_mask_2d = lambda p: seg
        fake_log = types.ModuleType("respan_uncaging_log")
        fake_log.parse_uncaging_records = lambda h: [Rec(flims[2])]
        saved = {k: sys.modules.get(k) for k in ("gui_roi_respan_seg_masks", "respan_uncaging_log")}
        sys.modules.update({"gui_roi_respan_seg_masks": fake, "respan_uncaging_log": fake_log})
        orig_ref = m.reference_roi
        m.reference_roi = lambda jdir, pick, xy: disk(shape, round(pick["head"][1]), round(pick["head"][2]), 4)
        try:
            df = pd.DataFrame({"filepath_without_number": "x", "group": "g", "nth_set_label": 0,
                               "phase": ["pre"] * 3 + ["unc"] + ["post"] * 2,
                               "after_align_save_path": tiff, "n_pre_frames": 3, "n_unc_frames": 2,
                               "n_post_frames": 2})
            s1 = m.apply_tracked_spine_rois(df, queue_root=str(q.root))
            assert s1.status.tolist() == ["written"], s1.to_dict("records")
            out = tifffile.imread(os.path.join(tif_dir, base + "_Spine_roi_mask.tif"))
            assert out.shape == (7,) + shape
            # Still spine: every GUI frame has the ROI at the first-pre raw head (30, 35).
            for i in range(7):
                cy, cx = ndi.center_of_mass(out[i])
                assert abs(cy - 30) < 0.5 and abs(cx - 35) < 0.5, (i, cy, cx)
            side = json.loads(open(os.path.join(tif_dir, base + "_Spine_roi_mask_source.json")).read())
            assert side["reference_frame"] == names[2] and side["lost_frames"] == []
            assert side["shaft_removed_px"] is None  # no shaft mask on disk yet
            # With a shaft mask on disk the overlapping pixels are removed (fresh mask -> replaceable).
            shaft = np.zeros((7,) + shape, np.uint8)
            shaft[:, 30:, :] = 1  # lower half of the spine disk
            tifffile.imwrite(os.path.join(tif_dir, base + "_DendriticShaft_roi_mask.tif"), shaft)
            s_sh = m.apply_tracked_spine_rois(df, queue_root=str(q.root))
            assert s_sh.status.tolist() == ["written"], s_sh.to_dict("records")
            out_sh = tifffile.imread(os.path.join(tif_dir, base + "_Spine_roi_mask.tif"))
            assert not (out_sh.astype(bool) & shaft.astype(bool)).any() and out_sh.any(axis=(1, 2)).all()
            side = json.loads(open(os.path.join(tif_dir, base + "_Spine_roi_mask_source.json")).read())
            # trimmed once in the reference frame: one count, same ROI size in every frame
            assert side["shaft_removed_px"] > 0 and side["sha1"] == m._sha(out_sh > 0)
            sizes = out_sh.reshape(len(out_sh), -1).sum(1)
            assert (sizes == sizes[0]).all(), sizes
            os.remove(os.path.join(tif_dir, base + "_DendriticShaft_roi_mask.tif"))
            m.apply_tracked_spine_rois(df, queue_root=str(q.root))
            out = tifffile.imread(os.path.join(tif_dir, base + "_Spine_roi_mask.tif"))
            # User edits the mask in the GUI -> a second run must keep it.
            edited = out.copy()
            edited[5] = 0
            tifffile.imwrite(os.path.join(tif_dir, base + "_Spine_roi_mask.tif"), edited)
            s2 = m.apply_tracked_spine_rois(df, queue_root=str(q.root))
            assert s2.status.tolist() == ["skip"] and "edited" in s2.reason[0]
            assert np.array_equal(tifffile.imread(os.path.join(tif_dir, base + "_Spine_roi_mask.tif")), edited)
        finally:
            m.reference_roi = orig_ref
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
            n += 1
    print(f"all {n} tests passed")
