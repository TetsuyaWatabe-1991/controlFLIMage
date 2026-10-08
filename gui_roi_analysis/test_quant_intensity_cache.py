"""Headless test: intensity-only quantification from the fast_mode intensity cache.

Unit test: load_photon_counts_cached returns exactly np.sum(bins) for synthetic data
(cache = 12 * sum / nAveFrame, several nAveFrame values).
--real: quantifies prepared set folders (combined_df_one_set.pkl + tif/, e.g. rebuilt
into a scratch folder) twice, decoding every .flim (use_intensity_cache=False) and from
the cache (True), and requires identical CSVs; prints both run times. Writing the
intensity cache is disabled, so session files are only read.

Run:
    python test_quant_intensity_cache.py
    python test_quant_intensity_cache.py --real <set_folder> [<set_folder> ...]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.dirname(HERE))
import flim_fast_io as ffi  # noqa: E402
import gui_roi_fast_simple as g  # noqa: E402


def test_counts_from_cache_are_exact():
    rng = np.random.default_rng(3)
    for n_ave in (1, 3, 5, 7, 16):
        counts = rng.poisson(4.0, size=(3, 1, 2, 16, 16, 8)).sum(-1)
        cache = ((12.0 * counts) / n_ave).astype(np.float32)  # what load_flim_intensity stores

        class _Acq:
            nAveFrame = n_ave

        class _Info:
            class State:
                Acq = _Acq

        orig = ffi.load_flim_intensity
        ffi.load_flim_intensity = lambda path, use_cache=True: (cache, _Info)
        try:
            got, _ = g.load_photon_counts_cached("x.flim")
        finally:
            ffi.load_flim_intensity = orig
        assert np.array_equal(got, counts.astype(np.float64)), n_ave


def real_check(folders: list[str]) -> None:
    ffi._save_cached_intensity = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as td:
        dfs = []
        for k, src in enumerate(folders):
            dst = os.path.join(td, f"set{k}")
            shutil.copytree(src, dst)
            df = pd.read_pickle(os.path.join(dst, "combined_df_one_set.pkl"))
            for col in ("after_align_save_path", "after_align_full_save_path"):
                df[col] = df[col].map(lambda p: os.path.join(dst, "tif", os.path.basename(str(p))) if isinstance(p, str) else p)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df["reject"] = 0
        g.save_drift_corrected_roi_masks(df, roi_types=["Spine", "DendriticShaft"])
        out = {}
        for use_cache in (False, True):
            p = os.path.join(td, f"q_{use_cache}.csv")
            t0 = time.time()
            g.quantify_intensity_from_flim(df, 2, 1, p, skip_lifetime_analysis=True,
                                           background_mode=g.BACKGROUND_MODE_MIP_P20,
                                           use_intensity_cache=use_cache)
            out[use_cache] = (pd.read_csv(p), time.time() - t0)
        a, b = out[False][0], out[True][0]
        assert a.shape == b.shape, (a.shape, b.shape)
        cols = [c for c in a.columns if c.endswith("_intensity")]
        assert cols and any(c.startswith("Spine_Ch1") for c in cols), cols
        for c in a.columns:
            if c in cols or a[c].dtype.kind in "fi":
                assert np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True, rtol=0, atol=1e-9), c
            else:
                assert (a[c].astype(str) == b[c].astype(str)).all(), c
        print(f"real: {len(a)} rows x {len(cols)} intensity columns identical (Ch1 and Ch2); "
              f"FLIM decode {out[False][1]:.1f} s vs intensity cache {out[True][1]:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", nargs="*", default=[])
    a = ap.parse_args()
    test_counts_from_cache_are_exact()
    print("PASS test_counts_from_cache_are_exact")
    if a.real:
        real_check(a.real)
