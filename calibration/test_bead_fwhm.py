# -*- coding: utf-8 -*-
"""Automated tests for bead FWHM core logic and headless save path."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

_CALIB = Path(__file__).resolve().parent
if str(_CALIB) not in sys.path:
    sys.path.insert(0, str(_CALIB))

from bead_fwhm_core import (  # noqa: E402
    MIN_LINE_PIXELS,
    PixelSizeUm,
    compute_fwhm,
    compute_projections,
    default_contrast,
    extract_line_profile,
    line_length_px,
    make_drawn_line,
    make_output_dir,
    make_synthetic_bead_volume,
    measure_center_lines_on_synthetic,
    measure_line,
    output_stem_prefix,
    run_self_test,
    snap_to_axis,
    theoretical_gaussian_fwhm_um,
)


def test_snap_auto_horizontal() -> None:
    x0, y0, x1, y1, orient = snap_to_axis(10, 20, 40, 22, mode="auto")
    assert orient == "horizontal"
    assert (x0, y0, x1, y1) == (10, 20, 40, 20)


def test_snap_auto_vertical() -> None:
    x0, y0, x1, y1, orient = snap_to_axis(10, 20, 12, 50, mode="auto")
    assert orient == "vertical"
    assert (x0, y0, x1, y1) == (10, 20, 10, 50)


def test_snap_forced_axes() -> None:
    _, _, x1, y1, orient = snap_to_axis(0, 0, 30, 40, mode="horizontal")
    assert orient == "horizontal"
    assert y1 == 0 and x1 == 30
    _, _, x1, y1, orient = snap_to_axis(0, 0, 30, 40, mode="vertical")
    assert orient == "vertical"
    assert x1 == 0 and y1 == 40


def test_projections_peak_at_bead_center() -> None:
    center = (20.0, 32.0, 32.0)
    volume = make_synthetic_bead_volume(center_zyx=center)
    z_proj, y_proj = compute_projections(volume)
    assert z_proj.shape == (64, 64)
    assert y_proj.shape == (41, 64)
    zy, zx = np.unravel_index(int(np.argmax(z_proj)), z_proj.shape)
    yy, yx = np.unravel_index(int(np.argmax(y_proj)), y_proj.shape)
    assert abs(zy - center[1]) <= 1
    assert abs(zx - center[2]) <= 1
    assert abs(yy - center[0]) <= 1
    assert abs(yx - center[2]) <= 1


def test_fwhm_known_gaussian() -> None:
    x = np.linspace(-8.0, 8.0, 161)
    sigma = 1.2
    y = 15.0 + 400.0 * np.exp(-0.5 * (x / sigma) ** 2)
    result = compute_fwhm(x, y, measured_axis="X", pixel_size_um=x[1] - x[0])
    expected = theoretical_gaussian_fwhm_um(sigma, 1.0)
    assert result.ok
    assert abs(result.fwhm_um - expected) / expected < 0.02


def test_fwhm_rejects_flat_profile() -> None:
    x = np.arange(20, dtype=float)
    y = np.ones_like(x) * 5.0
    result = compute_fwhm(x, y, measured_axis="X", pixel_size_um=1.0)
    assert not result.ok


def test_short_line_rejected() -> None:
    img = np.zeros((20, 20), dtype=float)
    line = make_drawn_line("z_proj", 5, 5, 6, 5, img.shape, "L1", "#fff", mode="horizontal")
    assert line_length_px(line) < MIN_LINE_PIXELS or line_length_px(line) == 2


def test_center_lines_match_theory() -> None:
    pixel_size = PixelSizeUm(z_um=0.50, y_um=0.10, x_um=0.10)
    sigma = (6.0, 3.0, 3.0)
    center = (20.0, 32.0, 32.0)
    volume = make_synthetic_bead_volume(
        center_zyx=center, sigma_zyx_px=sigma, background=25.0
    )
    _, _, measurements = measure_center_lines_on_synthetic(volume, pixel_size, center)
    by_label = {m.line.label: m for m in measurements}
    expected = {
        "L1": ("X", theoretical_gaussian_fwhm_um(sigma[2], pixel_size.x_um)),
        "L2": ("Y", theoretical_gaussian_fwhm_um(sigma[1], pixel_size.y_um)),
        "L3": ("X", theoretical_gaussian_fwhm_um(sigma[2], pixel_size.x_um)),
        "L4": ("Z", theoretical_gaussian_fwhm_um(sigma[0], pixel_size.z_um)),
    }
    assert len(measurements) == 4
    for label, (axis, exp) in expected.items():
        meas = by_label[label]
        assert meas.fwhm.measured_axis == axis
        assert meas.fwhm.ok
        rel = abs(meas.fwhm.fwhm_um - exp) / exp
        assert rel < 0.08, f"{label} rel err {rel:.3f}"


def test_profile_width_averaging() -> None:
    img = np.zeros((21, 21), dtype=float)
    img[10, :] = 10.0
    img[9, :] = 4.0
    img[11, :] = 4.0
    line = make_drawn_line(
        "z_proj", 0, 10, 20, 10, img.shape, "L1", "#f00", mode="horizontal", width_px=3
    )
    _, profile = extract_line_profile(img, line)
    np.testing.assert_allclose(profile, np.full(21, (10.0 + 4.0 + 4.0) / 3.0))


def test_self_test_writes_expected_files() -> None:
    with tempfile.TemporaryDirectory(prefix="bead_fwhm_") as tmp:
        result = run_self_test(tmp, rel_tol=0.08)
        out = Path(result["saved"]["out_dir"])
        prefix = output_stem_prefix("synthetic_bead")
        for name in (
            f"{prefix}z_projection_with_lines.png",
            f"{prefix}y_projection_with_lines.png",
            f"{prefix}intensity_profiles.png",
            f"{prefix}combined.png",
            f"{prefix}fwhm_results.csv",
            f"{prefix}intensity_profiles.csv",
        ):
            path = out / name
            assert path.is_file(), name
            assert path.stat().st_size > 0, name
        with result["saved"]["fwhm_csv"].open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4
        assert result["n_profile_rows"] > 20
        print(f"  self-test output: {out}")
        print(f"  FWHM CSV rows: {len(rows)}")
        print(f"  profile CSV rows: {result['n_profile_rows']}")


def test_make_output_dir_defaults_to_flim_folder() -> None:
    with tempfile.TemporaryDirectory(prefix="bead_fwhm_dir_") as tmp:
        flim = Path(tmp) / "bead_001.flim"
        flim.write_bytes(b"placeholder")
        out = make_output_dir(None, flim)
        assert out.resolve() == Path(tmp).resolve()
        override = Path(tmp) / "override"
        out2 = make_output_dir(override, flim)
        assert out2.resolve() == override.resolve()
        assert override.is_dir()


def test_gui_save_next_to_flim() -> None:
    """Save plots/CSV into the folder that contains the .flim file."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from bead_fwhm_gui import BeadFwhmWindow, load_synthetic_demo

    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="bead_fwhm_nextto_") as tmp:
        flim_dir = Path(tmp)
        flim_path = flim_dir / "0,2beads_067.flim"
        loaded = load_synthetic_demo()
        loaded.path = str(flim_path)
        win = BeadFwhmWindow(save_dir=None)
        win._apply_loaded(loaded)
        meas = win.add_line_from_pixels("z_proj", 0, 32, 63, 32, mode="horizontal")
        assert meas is not None and meas.fwhm.ok
        saved = win.export_outputs()
        assert saved["out_dir"].resolve() == flim_dir.resolve()
        prefix = output_stem_prefix(flim_path.name)
        assert saved["fwhm_csv"].name == f"{prefix}fwhm_results.csv"
        assert saved["combined"].parent.resolve() == flim_dir.resolve()
        assert saved["fwhm_csv"].is_file()
        print(f"  saved next to flim: {saved['fwhm_csv']}")
        win.close()
        del win
        if app is not QApplication.instance():
            pass


def test_contrast_defaults() -> None:
    img = np.array([[1.0, 5.0], [2.0, 10.0]])
    vmin, vmax = default_contrast(img)
    assert vmin == 1.0
    assert vmax == 10.0


def test_measure_line_uses_pixel_size() -> None:
    img = np.zeros((32, 64), dtype=float)
    x = np.arange(64)
    img[16, :] = 20 + 200 * np.exp(-0.5 * ((x - 32) / 4.0) ** 2)
    line = make_drawn_line(
        "z_proj", 0, 16, 63, 16, img.shape, "L1", "#0f0", mode="horizontal"
    )
    meas = measure_line(img, line, PixelSizeUm(z_um=1.0, y_um=0.2, x_um=0.2))
    assert meas.fwhm.ok
    expected = theoretical_gaussian_fwhm_um(4.0, 0.2)
    rel = abs(meas.fwhm.fwhm_um - expected) / expected
    assert rel < 0.08, rel


def test_gui_programmatic_lines_and_save() -> None:
    """Headless GUI: synthetic volume, two axis-parallel lines, save plots."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from bead_fwhm_gui import BeadFwhmWindow, load_synthetic_demo

    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="bead_fwhm_gui_") as tmp:
        win = BeadFwhmWindow(save_dir=Path(tmp))
        win._apply_loaded(load_synthetic_demo())
        win.z_contrast.min_spin.setValue(0.0)
        win.z_contrast.max_spin.setValue(800.0)
        win.y_contrast.min_spin.setValue(0.0)
        win.y_contrast.max_spin.setValue(800.0)
        m1 = win.add_line_from_pixels("z_proj", 0, 32, 63, 32, mode="horizontal")
        m2 = win.add_line_from_pixels("y_proj", 32, 0, 32, 40, mode="vertical")
        assert m1 is not None and m1.fwhm.ok
        assert m2 is not None and m2.fwhm.ok
        assert m1.fwhm.measured_axis == "X"
        assert m2.fwhm.measured_axis == "Z"
        assert win.table.rowCount() == 2
        saved = win.export_outputs(Path(tmp) / "gui_out")
        for key in ("z_projection", "y_projection", "intensity_profiles", "combined", "fwhm_csv"):
            assert saved[key].is_file()
            assert saved[key].stat().st_size > 0
        with saved["fwhm_csv"].open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        print(f"  GUI save dir: {saved['out_dir']}")
        print(f"  GUI FWHM CSV rows: {len(rows)}")
        win.close()
        del win
        if app is not QApplication.instance():
            pass


def test_runner_resolve_requires_path() -> None:
    from run_beads_fwhm_gui import resolve_flim_path

    try:
        resolve_flim_path("   ")
    except ValueError as exc:
        assert "FLIM_PATH" in str(exc)
        return
    raise AssertionError("expected ValueError for empty FLIM_PATH")


def test_runner_resolve_missing_file() -> None:
    from run_beads_fwhm_gui import resolve_flim_path

    try:
        resolve_flim_path(r"C:\this_file_does_not_exist_bead_fwhm.flim")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_runner_resolve_existing_file_and_quotes() -> None:
    from run_beads_fwhm_gui import resolve_flim_path

    handle = tempfile.NamedTemporaryFile(suffix=".flim", delete=False)
    path = handle.name
    handle.write(b"placeholder")
    handle.close()
    try:
        resolved = resolve_flim_path(path)
        assert resolved.is_file()
        quoted = resolve_flim_path(f'"{path}"')
        assert quoted == resolved
    finally:
        Path(path).unlink(missing_ok=True)


def test_launch_window_headless_without_file() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from bead_fwhm_gui import launch_window

    code = launch_window(show=False)
    assert code == 0


def main() -> int:
    tests = [
        test_snap_auto_horizontal,
        test_snap_auto_vertical,
        test_snap_forced_axes,
        test_projections_peak_at_bead_center,
        test_fwhm_known_gaussian,
        test_fwhm_rejects_flat_profile,
        test_short_line_rejected,
        test_center_lines_match_theory,
        test_profile_width_averaging,
        test_contrast_defaults,
        test_measure_line_uses_pixel_size,
        test_self_test_writes_expected_files,
        test_make_output_dir_defaults_to_flim_folder,
        test_gui_programmatic_lines_and_save,
        test_gui_save_next_to_flim,
        test_runner_resolve_requires_path,
        test_runner_resolve_missing_file,
        test_runner_resolve_existing_file_and_quotes,
        test_launch_window_headless_without_file,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL: {fn.__name__}: {exc}")
    print(f"Done: {len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
