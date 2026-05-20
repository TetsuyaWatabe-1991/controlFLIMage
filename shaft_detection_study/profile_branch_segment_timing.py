# -*- coding: utf-8 -*-
"""Profile the branch-segment pipeline end-to-end.

Synthetic mode is local-only (no network FLIM read) so we can see *where* the time goes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(_STUDY.parent))


def _make_synthetic(shape=(24, 256, 256)) -> tuple[np.ndarray, float, float]:
    """Synthetic stack: a few elongated 'dendrites' on dark background."""
    rng = np.random.default_rng(0)
    zyx = rng.normal(loc=80.0, scale=10.0, size=shape).astype(np.float32)
    Z, Y, X = shape
    yy, xx = np.indices((Y, X))
    for z0, y0, x0, dy, dx in [
        (Z // 2, 60, 50, 0.1, 1.0),
        (Z // 2, 120, 60, 0.0, 1.0),
        (Z // 2 + 1, 180, 70, -0.1, 1.0),
        (Z // 2 - 1, 90, 100, 0.0, 1.0),
    ]:
        line = (yy - (y0 + dy * (xx - x0))) ** 2 < 9.0
        line &= (xx > x0) & (xx < x0 + 150)
        for dz in range(-1, 2):
            zi = max(0, min(Z - 1, z0 + dz))
            zyx[zi][line] += 400.0
    # add some spine bumps
    for _ in range(80):
        z, y, x = rng.integers([0, 0, 0], shape)
        zyx[max(0, z - 1) : z + 2, max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] += 200
    return zyx, 0.267, 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["synthetic", "flim"],
        default="synthetic",
    )
    parser.add_argument("--flim", type=str, default=None)
    args = parser.parse_args()

    from shaft_detection_study.dendrite_branch_score_core import (  # noqa: E402
        dendrite_width_transform,
        label_skeleton_branches,
        score_branch_point_set,
        select_spaced_branches,
        _detect_puncta_near_branch_pts,
    )
    from lowmag_mushroom_branch_finder import (  # noqa: E402
        build_dendrite_and_protrusion_maps,
        pick_best_point_on_branch,
    )

    timings: list[tuple[str, float]] = []

    def tick(name: str, t0: float) -> None:
        dt = time.perf_counter() - t0
        timings.append((name, dt))
        print(f"  [{dt:7.3f}s] {name}", flush=True)

    print(f"\n=== mode={args.mode} ===", flush=True)

    if args.mode == "synthetic":
        zyx, x_um, z_um = _make_synthetic()
        print(f"  synthetic ZYX shape={zyx.shape}, xy={x_um}, z={z_um}", flush=True)
    else:
        from shaft_detection_study.dendrite_branch_score_core import load_flim_zyx
        if not args.flim:
            raise SystemExit("--flim required for mode=flim")
        t0 = time.perf_counter()
        zyx, x_um, _, z_um = load_flim_zyx(args.flim)
        tick(f"load_flim shape={zyx.shape}", t0)

    t0 = time.perf_counter()
    _, dendrite_mask, protrusion, skeleton_3d, _, _, _ = (
        build_dendrite_and_protrusion_maps(zyx, x_um, min_skeleton_length_um=20.0)
    )
    tick("build_maps_skeleton (20um prune)", t0)

    t0 = time.perf_counter()
    width_um_vol = dendrite_width_transform(dendrite_mask, x_um)
    tick("width_edt_once", t0)

    t0 = time.perf_counter()
    labeled_sk, n_branch = label_skeleton_branches(skeleton_3d)
    tick(f"label_branches (n={n_branch})", t0)

    t0 = time.perf_counter()
    sk_z, sk_y, sk_x = np.where(labeled_sk > 0)
    sk_labels = labeled_sk[sk_z, sk_y, sk_x]
    order = np.argsort(sk_labels, kind="stable")
    sk_z = sk_z[order]
    sk_y = sk_y[order]
    sk_x = sk_x[order]
    sk_labels = sk_labels[order]
    boundaries = np.searchsorted(sk_labels, np.arange(1, n_branch + 2))
    tick("group_voxels_by_label", t0)

    t0 = time.perf_counter()
    qualifying = []
    for bid in range(1, n_branch + 1):
        i0, i1 = int(boundaries[bid - 1]), int(boundaries[bid])
        n_pts = i1 - i0
        path_est = n_pts * x_um * 0.85
        if n_pts < 8 or path_est < 20 * 0.75:
            continue
        z = sk_z[i0:i1]
        y = sk_y[i0:i1]
        x = sk_x[i0:i1]
        widths = width_um_vol[z, y, x]
        mw = float(widths.mean())
        if mw <= 0.6 or mw > 1.0:
            continue
        branch_pts = np.stack([z, y, x], axis=1).astype(np.int64)
        qualifying.append(
            {
                "branch_id": bid,
                "branch_pts": branch_pts,
                "widths_um": widths.astype(np.float64),
                "path_length_um": path_est,
                "rep_zyx": branch_pts[n_pts // 2],
            }
        )
    tick(f"filter_qualifying (n={len(qualifying)})", t0)

    t0 = time.perf_counter()
    selected = select_spaced_branches(qualifying, 20, 25.0, x_um, z_um)
    tick(f"select_spaced (n={len(selected)})", t0)

    t0 = time.perf_counter()
    puncta = _detect_puncta_near_branch_pts(
        protrusion, dendrite_mask, x_um, [s["branch_pts"] for s in selected]
    )
    tick(f"detect_puncta_local (n={len(puncta)})", t0)

    t0 = time.perf_counter()
    for info in selected:
        pick_best_point_on_branch(
            info["branch_pts"], protrusion, puncta, x_um, z_um, local_radius_um=7.5
        )
    tick("pick_best_point loop", t0)

    t0 = time.perf_counter()
    seg_radius_um = 7.5
    spacing = np.array([z_um, x_um, x_um])
    for info in selected:
        rep, _ = pick_best_point_on_branch(
            info["branch_pts"], protrusion, puncta, x_um, z_um, local_radius_um=seg_radius_um
        )
        b_phys = info["branch_pts"].astype(np.float64) * spacing
        d = np.linalg.norm(b_phys - rep.astype(np.float64) * spacing, axis=1)
        pts = info["branch_pts"][d <= seg_radius_um]
        if len(pts) < 3:
            continue
        score_branch_point_set(
            pts, dendrite_mask, protrusion, puncta, x_um, z_um, width_um_vol=width_um_vol
        )
    tick("score loop", t0)

    print(f"\n  TOTAL: {sum(d for _, d in timings):.3f}s", flush=True)


if __name__ == "__main__":
    main()
