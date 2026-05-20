# -*- coding: utf-8 -*-
"""
Score skeleton branches for low-mag dendrite candidate picking (no DeepD3).

High score: relatively uniform (smooth) shaft + clear spine protrusions on Z-MIP.
Low score: irregular shaft width (thick/thin along branch), few spines.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree

_STUDY_DIR = Path(__file__).resolve().parent
_CONTROLFLIMAGE = _STUDY_DIR.parent
if str(_CONTROLFLIMAGE) not in sys.path:
    sys.path.insert(0, str(_CONTROLFLIMAGE))

from lowmag_mushroom_branch_finder import (  # noqa: E402
    branch_corridor_mask,
    build_dendrite_and_protrusion_maps,
    detect_puncta_near_dendrite,
    label_skeleton_branches,
    mip_branch_roughness,
    physical_distance_um,
    pick_best_point_on_branch,
)
from FLIMageAlignment import get_xyz_pixel_um  # noqa: E402
from FLIMageFileReader2 import FileReader  # noqa: E402

# Exclude branches whose mean shaft width is outside this range (um).
DEFAULT_MIN_BRANCH_WIDTH_UM = 0.3
DEFAULT_MAX_BRANCH_WIDTH_UM = 1.1


def branch_width_ok(
    widths_um: np.ndarray,
    min_width_um: float,
    max_width_um: float,
) -> tuple[bool, str]:
    """
  Return False if mean width <= min_width_um or mean width > max_width_um.
    """
    mean_w = float(np.mean(widths_um))
    if mean_w <= min_width_um:
        return False, f"mean_width {mean_w:.3f} um <= {min_width_um} um"
    if mean_w > max_width_um:
        return False, f"mean_width {mean_w:.3f} um > {max_width_um} um"
    return True, ""


def load_flim_zyx(flim_path: str) -> tuple[np.ndarray, float, float, float]:
    iminfo = FileReader()
    iminfo.read_imageFile(flim_path, True)
    zyx = np.array(iminfo.image).sum(axis=(1, 2, 5))
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    return np.asarray(zyx, dtype=np.float32), float(x_um), float(y_um), float(z_um)


def _pick_rep_point_fast(
    branch_pts: np.ndarray,
    protrusion: np.ndarray,
    xy_um: float,
    z_um: float,
    local_radius_um: float,
    *,
    max_candidates: int = 60,
) -> np.ndarray:
    """Find rep point with high local protrusion sum, sampling at most ~60 voxels.

    Uses a uniform_filter pre-pass on protrusion so each candidate lookup is O(1).
    """
    from scipy.ndimage import uniform_filter

    n = len(branch_pts)
    if n == 0:
        return np.zeros(3, dtype=int)
    if n == 1:
        return branch_pts[0].astype(int)

    ry = max(1, int(round(local_radius_um / xy_um)))
    rz = max(1, int(round(local_radius_um / max(z_um, 0.01))))
    size = (2 * rz + 1, 2 * ry + 1, 2 * ry + 1)

    z0 = max(0, int(branch_pts[:, 0].min()) - rz)
    z1 = min(protrusion.shape[0], int(branch_pts[:, 0].max()) + rz + 1)
    y0 = max(0, int(branch_pts[:, 1].min()) - ry)
    y1 = min(protrusion.shape[1], int(branch_pts[:, 1].max()) + ry + 1)
    x0 = max(0, int(branch_pts[:, 2].min()) - ry)
    x1 = min(protrusion.shape[2], int(branch_pts[:, 2].max()) + ry + 1)
    sub = protrusion[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
    if sub.size == 0:
        return branch_pts[n // 2].astype(int)
    smoothed = uniform_filter(sub, size=size, mode="constant")

    step = max(1, n // max_candidates)
    cand = branch_pts[::step]
    zi = cand[:, 0] - z0
    yi = cand[:, 1] - y0
    xi = cand[:, 2] - x0
    zi = np.clip(zi, 0, smoothed.shape[0] - 1)
    yi = np.clip(yi, 0, smoothed.shape[1] - 1)
    xi = np.clip(xi, 0, smoothed.shape[2] - 1)
    vals = smoothed[zi, yi, xi]
    best = cand[int(np.argmax(vals))].astype(int)
    return best


def _detect_puncta_near_branch_pts(
    protrusion: np.ndarray,
    dendrite_mask: np.ndarray,
    xy_um: float,
    branch_pts_list: list[np.ndarray],
    margin_um: float = 8.0,
) -> np.ndarray:
    """Peak detection only in a tight ROI around selected branches (fast)."""
    from skimage.feature import peak_local_max

    if not branch_pts_list:
        return np.zeros((0, 3), dtype=int)
    all_pts = np.vstack(branch_pts_list)
    z0, y0, x0 = all_pts.min(axis=0).astype(int)
    z1, y1, x1 = all_pts.max(axis=0).astype(int) + 1
    my = max(2, int(round(margin_um / xy_um)))
    mz = max(1, int(round(margin_um / max(xy_um, 0.01))))
    z0, y0, x0 = max(0, z0 - mz), max(0, y0 - my), max(0, x0 - my)
    z1 = min(protrusion.shape[0], z1 + mz)
    y1 = min(protrusion.shape[1], y1 + my)
    x1 = min(protrusion.shape[2], x1 + my)

    min_dist_pix = max(1, int(round(0.8 / xy_um)))
    pos = protrusion[z0:z1, y0:y1, x0:x1]
    pos = pos[pos > 0]
    if len(pos) == 0:
        return np.zeros((0, 3), dtype=int)
    thresh = float(np.percentile(pos, 70.0))

    puncta: list[list[int]] = []
    for z in range(z0, z1):
        sl = protrusion[z, y0:y1, x0:x1]
        if sl.max() <= thresh:
            continue
        peaks = peak_local_max(
            sl,
            min_distance=min_dist_pix,
            threshold_abs=thresh,
            exclude_border=True,
        )
        for y, x in peaks:
            if dendrite_mask[z, y0 + y, x0 + x]:
                puncta.append([z, y0 + int(y), x0 + int(x)])
    return np.array(puncta, dtype=int) if puncta else np.zeros((0, 3), dtype=int)


def dendrite_width_transform(dendrite_mask: np.ndarray, xy_um: float) -> np.ndarray:
    """Precompute EDT on dendrite mask once (um at skeleton samples)."""
    return distance_transform_edt(dendrite_mask) * xy_um * 2.0


def shaft_width_profile_um(
    branch_pts: np.ndarray,
    dendrite_mask: np.ndarray,
    xy_um: float,
    *,
    width_um_vol: np.ndarray | None = None,
) -> np.ndarray:
    """Local shaft width (um) at each skeleton voxel via distance transform."""
    if width_um_vol is None:
        width_um_vol = dendrite_width_transform(dendrite_mask, xy_um)
    z, y, x = branch_pts.T.astype(int)
    return width_um_vol[z, y, x].astype(np.float64)


def shaft_uniformity_metrics(widths_um: np.ndarray) -> dict[str, float]:
    """
    smoothness in [0,1]: 1 = uniform diameter along branch.
    width_cv / width_range_ratio increase when shaft is irregular.
    """
    if len(widths_um) < 3:
        return {
            "mean_width_um": float(np.mean(widths_um)) if len(widths_um) else 0.0,
            "width_cv": 1.0,
            "width_range_ratio": 1.0,
            "smoothness": 0.0,
        }
    mean_w = float(np.mean(widths_um))
    std_w = float(np.std(widths_um))
    width_cv = std_w / (mean_w + 1e-6)
    width_range_ratio = float((widths_um.max() - widths_um.min()) / (mean_w + 1e-6))
    smoothness = float(np.exp(-2.0 * width_cv) * np.exp(-1.2 * width_range_ratio))
    return {
        "mean_width_um": mean_w,
        "min_width_um": float(np.min(widths_um)),
        "max_width_um": float(np.max(widths_um)),
        "width_cv": width_cv,
        "width_range_ratio": width_range_ratio,
        "smoothness": smoothness,
    }


def spine_signal_metrics(
    branch_pts: np.ndarray,
    corridor: np.ndarray,
    protrusion: np.ndarray,
    dendrite_mask: np.ndarray,
    puncta_zyx: np.ndarray,
    xy_um: float,
    z_um: float,
    tube_radius_um: float,
) -> dict[str, float]:
    sk_len = max(len(branch_pts), 1)
    prot_vals = protrusion[corridor]
    protrusion_density = float(prot_vals.sum() / sk_len)
    protrusion_mean = float(prot_vals.mean()) if prot_vals.size else 0.0

    if len(puncta_zyx) > 0:
        p_phys = puncta_zyx * np.array([z_um, xy_um, xy_um])
        b_phys = branch_pts * np.array([z_um, xy_um, xy_um])
        dists, _ = cKDTree(b_phys).query(p_phys, k=1)
        puncta_count = int(np.sum(dists <= tube_radius_um))
    else:
        puncta_count = 0

    dendrite_mip = dendrite_mask.max(axis=0)
    corridor_mip = corridor.max(axis=0)
    roughness = mip_branch_roughness(dendrite_mip, corridor_mip)
    puncta_per_len = puncta_count / sk_len

    # Normalize-ish spine component (higher = more blobby spines on branch)
    spine_raw = (
        protrusion_density
        + 2.5 * puncta_per_len
        + 0.12 * roughness
        + 0.4 * protrusion_mean
    )
    return {
        "protrusion_density": protrusion_density,
        "protrusion_mean": protrusion_mean,
        "puncta_count": float(puncta_count),
        "puncta_per_len": float(puncta_per_len),
        "roughness": roughness,
        "spine_signal": spine_raw,
    }


def score_dendrite_branch(
    branch_id: int,
    labeled_skeleton: np.ndarray,
    protrusion: np.ndarray,
    dendrite_mask: np.ndarray,
    puncta_zyx: np.ndarray,
    xy_um: float,
    z_um: float,
    *,
    tube_radius_um: float = 3.5,
    min_branch_width_um: float = DEFAULT_MIN_BRANCH_WIDTH_UM,
    max_branch_width_um: float = DEFAULT_MAX_BRANCH_WIDTH_UM,
    min_skeleton_length_pix: int = 8,
    verbose: bool = False,
) -> dict[str, Any] | None:
    corridor, branch_pts = branch_corridor_mask(
        labeled_skeleton, branch_id, xy_um, z_um, tube_radius_um
    )
    if len(branch_pts) < min_skeleton_length_pix:
        return None

    widths_um = shaft_width_profile_um(branch_pts, dendrite_mask, xy_um)
    ok, reason = branch_width_ok(widths_um, min_branch_width_um, max_branch_width_um)
    if not ok:
        if verbose:
            print(f"  skip branch {branch_id}: {reason}")
        return None

    uniform = shaft_uniformity_metrics(widths_um)
    spine = spine_signal_metrics(
        branch_pts,
        corridor,
        protrusion,
        dendrite_mask,
        puncta_zyx,
        xy_um,
        z_um,
        tube_radius_um,
    )

    # Combined: spines on a smooth shaft win; irregular shaft is down-weighted
    branch_score = spine["spine_signal"] * uniform["smoothness"]

    return {
        "branch_id": branch_id,
        "branch_score": branch_score,
        "skeleton_length_pix": len(branch_pts),
        "branch_pts": branch_pts,
        **spine,
        **uniform,
    }


def branch_path_length_um(branch_pts: np.ndarray, xy_um: float, z_um: float) -> float:
    """Longest geodesic along 26-connected skeleton voxels (um)."""
    ordered, cumdist = order_branch_points_longest_path(branch_pts, xy_um, z_um)
    if len(cumdist) == 0:
        return 0.0
    return float(cumdist[-1])


def order_branch_points_longest_path(
    branch_pts: np.ndarray,
    xy_um: float,
    z_um: float,
    *,
    max_points: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Order skeleton voxels along the longest path (graph diameter) and return
    cumulative arc length in um for each point.
    """
    n = len(branch_pts)
    if n == 0:
        return branch_pts, np.zeros(0, dtype=np.float64)
    if n == 1:
        return branch_pts, np.zeros(1, dtype=np.float64)

    if n > max_points:
        step = max(1, n // max_points)
        branch_pts = branch_pts[::step].copy()

    spacing = np.array([z_um, xy_um, xy_um], dtype=np.float64)
    phys = branch_pts.astype(np.float64) * spacing
    tree = cKDTree(branch_pts)
    pairs = tree.query_pairs(r=1.75)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j in pairs:
        w = float(np.linalg.norm(phys[i] - phys[j]))
        adj[i].append((j, w))
        adj[j].append((i, w))

    def farthest(start: int) -> tuple[int, float, dict[int, float]]:
        from heapq import heappop, heappush

        INF = float("inf")
        dist = [INF] * n
        dist[start] = 0.0
        pq: list[tuple[float, int]] = [(0.0, start)]
        while pq:
            d_u, u = heappop(pq)
            if d_u > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d_u + w
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(pq, (nd, v))
        reachable = {i: d for i, d in enumerate(dist) if d < INF}
        end = max(reachable, key=reachable.get)
        return end, reachable[end], reachable

    def path_from_to(start: int, end: int) -> list[int]:
        from collections import deque

        prev = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            if u == end:
                break
            for v, _w in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if end not in prev:
            return [start]
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    a, _, _ = farthest(0)
    b, total_len, _ = farthest(a)
    path_idx = path_from_to(a, b)
    ordered = branch_pts[np.asarray(path_idx, dtype=int)]
    seg_lens = [
        0.0
        if i == 0
        else float(np.linalg.norm(phys[path_idx[i]] - phys[path_idx[i - 1]]))
        for i in range(len(path_idx))
    ]
    cumdist = np.cumsum(seg_lens)
    return ordered, cumdist


def split_path_into_segments(
    ordered_pts: np.ndarray,
    cumdist_um: np.ndarray,
    segment_length_um: float,
    *,
    min_segment_length_um: float = 5.0,
) -> list[dict[str, Any]]:
    """Split ordered centerline into arc-length bins (default 15 um)."""
    if len(ordered_pts) == 0:
        return []
    total = float(cumdist_um[-1]) if len(cumdist_um) else 0.0
    if total <= 0:
        return [
            {
                "segment_index": 0,
                "arc_start_um": 0.0,
                "arc_end_um": 0.0,
                "branch_pts": ordered_pts.copy(),
            }
        ]

    n_seg = max(1, int(np.ceil(total / segment_length_um)))
    segments: list[dict[str, Any]] = []
    seg_idx = 0
    for i in range(n_seg):
        seg_start = i * segment_length_um
        seg_end = min((i + 1) * segment_length_um, total)
        span = seg_end - seg_start
        if span < min_segment_length_um and i < n_seg - 1:
            continue
        mask = (cumdist_um >= seg_start - 1e-6) & (cumdist_um <= seg_end + 1e-6)
        pts = ordered_pts[mask]
        if len(pts) < 2 and i < n_seg - 1:
            continue
        if len(pts) == 0:
            continue
        segments.append(
            {
                "segment_index": seg_idx,
                "arc_start_um": float(seg_start),
                "arc_end_um": float(seg_end),
                "branch_pts": pts,
            }
        )
        seg_idx += 1
    return segments


def pick_representative_segment(
    ordered_pts: np.ndarray,
    cumdist_um: np.ndarray,
    protrusion: np.ndarray,
    segment_length_um: float,
    *,
    slide_step_um: float = 5.0,
) -> dict[str, Any] | None:
    """
    Pick one 15 um window with highest skeleton protrusion sum (fast proxy for spiny).
    """
    if len(ordered_pts) < 2:
        return None
    total = float(cumdist_um[-1])
    seg_len = min(segment_length_um, total)
    if total <= seg_len + 1e-6:
        return {
            "segment_index": 0,
            "arc_start_um": 0.0,
            "arc_end_um": total,
            "branch_pts": ordered_pts.copy(),
        }

    best_prot = -1.0
    best: dict[str, Any] | None = None
    step = max(2.0, slide_step_um)
    arc_start = 0.0
    while arc_start <= total - seg_len + 1e-6:
        arc_end = arc_start + seg_len
        mask = (cumdist_um >= arc_start - 1e-6) & (cumdist_um <= arc_end + 1e-6)
        pts = ordered_pts[mask]
        if len(pts) >= 2:
            prot_sum = float(protrusion[tuple(pts.T.astype(int))].sum())
            if prot_sum > best_prot:
                best_prot = prot_sum
                best = {
                    "segment_index": 0,
                    "arc_start_um": float(arc_start),
                    "arc_end_um": float(arc_end),
                    "branch_pts": pts,
                }
        arc_start += step
    return best


def select_spaced_branches(
    branch_infos: list[dict[str, Any]],
    max_num: int,
    min_spacing_um: float,
    x_um: float,
    z_um: float,
) -> list[dict[str, Any]]:
    """Greedy pick up to max_num branches spaced in 3D (rep point = best spine locus)."""
    ordered = sorted(branch_infos, key=lambda b: b["path_length_um"], reverse=True)
    selected: list[dict[str, Any]] = []
    for info in ordered:
        if len(selected) >= max_num:
            break
        rep = info["rep_zyx"]
        if all(
            physical_distance_um(rep, s["rep_zyx"], x_um, z_um) >= min_spacing_um
            for s in selected
        ):
            selected.append(info)
    return selected


def corridor_from_branch_points(
    branch_pts: np.ndarray,
    template_shape: tuple[int, ...],
    xy_um: float,
    z_um: float,
    tube_radius_um: float,
) -> np.ndarray:
    """Dilation only inside the bounding box of branch_pts (O(bbox), not O(volume))."""
    vol = np.zeros(template_shape, dtype=bool)
    if len(branch_pts) == 0:
        return vol
    ry = max(1, int(round(tube_radius_um / xy_um)))
    rz = max(1, int(round(tube_radius_um / z_um)))
    pts = branch_pts.astype(int)
    z_min = max(0, int(pts[:, 0].min()) - rz - 1)
    z_max = min(template_shape[0], int(pts[:, 0].max()) + rz + 2)
    y_min = max(0, int(pts[:, 1].min()) - ry - 1)
    y_max = min(template_shape[1], int(pts[:, 1].max()) + ry + 2)
    x_min = max(0, int(pts[:, 2].min()) - ry - 1)
    x_max = min(template_shape[2], int(pts[:, 2].max()) + ry + 2)
    sub_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    sub = np.zeros(sub_shape, dtype=bool)
    sub[pts[:, 0] - z_min, pts[:, 1] - y_min, pts[:, 2] - x_min] = True
    sub = binary_dilation(
        sub, structure=np.ones((2 * rz + 1, 2 * ry + 1, 2 * ry + 1))
    )
    vol[z_min:z_max, y_min:y_max, x_min:x_max] = sub
    return vol


def score_branch_point_set(
    branch_pts: np.ndarray,
    dendrite_mask: np.ndarray,
    protrusion: np.ndarray,
    puncta_zyx: np.ndarray,
    xy_um: float,
    z_um: float,
    *,
    tube_radius_um: float = 3.5,
    width_um_vol: np.ndarray | None = None,
) -> dict[str, Any] | None:
    if len(branch_pts) < 2:
        return None
    corridor = corridor_from_branch_points(
        branch_pts, dendrite_mask.shape, xy_um, z_um, tube_radius_um
    )
    widths_um = shaft_width_profile_um(
        branch_pts, dendrite_mask, xy_um, width_um_vol=width_um_vol
    )
    uniform = shaft_uniformity_metrics(widths_um)
    spine = spine_signal_metrics(
        branch_pts,
        corridor,
        protrusion,
        dendrite_mask,
        puncta_zyx,
        xy_um,
        z_um,
        tube_radius_um,
    )
    segment_score = spine["spine_signal"] * uniform["smoothness"]
    return {
        "branch_score": segment_score,
        "segment_score": segment_score,
        "skeleton_length_pix": len(branch_pts),
        **spine,
        **uniform,
    }


def score_all_branches(
    flim_path: str,
    *,
    dendrite_percentile: float = 94.0,
    soma_percentile: float = 99.0,
    threshold_area_um2: float = 10.0,
    shaft_opening_um: float = 1.2,
    tube_radius_um: float = 3.5,
    min_branch_width_um: float = DEFAULT_MIN_BRANCH_WIDTH_UM,
    max_branch_width_um: float = DEFAULT_MAX_BRANCH_WIDTH_UM,
    min_skeleton_length_pix: int = 12,
    min_skeleton_length_um: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    zyx, x_um, y_um, z_um = load_flim_zyx(flim_path)
    build_kw: dict[str, Any] = dict(
        dendrite_percentile=dendrite_percentile,
        soma_percentile=soma_percentile,
        threshold_area_um2=threshold_area_um2,
        shaft_opening_um=shaft_opening_um,
    )
    if min_skeleton_length_um is not None and min_skeleton_length_um > 0:
        build_kw["min_skeleton_length_um"] = min_skeleton_length_um
    denoised, dendrite_mask, protrusion, skeleton_3d, _, _, _ = build_dendrite_and_protrusion_maps(
        zyx,
        x_um,
        **build_kw,
    )
    puncta_zyx = detect_puncta_near_dendrite(protrusion, dendrite_mask, x_um)
    labeled_sk, n_branch = label_skeleton_branches(skeleton_3d)

    candidates: list[dict[str, Any]] = []
    for branch_id in range(1, n_branch + 1):
        scored = score_dendrite_branch(
            branch_id,
            labeled_sk,
            protrusion,
            dendrite_mask,
            puncta_zyx,
            x_um,
            z_um,
            tube_radius_um=tube_radius_um,
            min_branch_width_um=min_branch_width_um,
            max_branch_width_um=max_branch_width_um,
            min_skeleton_length_pix=min_skeleton_length_pix,
            verbose=verbose,
        )
        if scored is None:
            continue
        zyx_pt, _ = pick_best_point_on_branch(
            scored["branch_pts"],
            protrusion,
            puncta_zyx,
            x_um,
            z_um,
            local_radius_um=10.0,
        )
        candidates.append(
            {
                "branch_id": branch_id,
                "zyx": zyx_pt,
                **{k: v for k, v in scored.items() if k != "branch_pts"},
            }
        )

    candidates.sort(key=lambda c: c["branch_score"], reverse=True)
    return {
        "flim_path": flim_path,
        "zyx": zyx,
        "x_um": x_um,
        "y_um": y_um,
        "z_um": z_um,
        "skeleton_3d": skeleton_3d,
        "protrusion": protrusion,
        "dendrite_mask": dendrite_mask,
        "puncta_zyx": puncta_zyx,
        "candidates": candidates,
        "n_branches": n_branch,
        "min_branch_width_um": min_branch_width_um,
        "max_branch_width_um": max_branch_width_um,
    }


def score_all_branch_segments(
    flim_path: str,
    *,
    min_branch_width_um: float = 0.6,
    max_branch_width_um: float = 1.0,
    min_branch_length_um: float = 20.0,
    segment_length_um: float = 15.0,
    max_branches: int = 20,
    min_spacing_um: float = 25.0,
    dendrite_percentile: float = 94.0,
    soma_percentile: float = 99.0,
    threshold_area_um2: float = 10.0,
    shaft_opening_um: float = 1.2,
    tube_radius_um: float = 3.5,
    overlap_um: float = 5.0,
    verbose: bool = True,
    profile: bool = False,
) -> dict[str, Any]:
    """
    Sliding-window scoring: split each qualifying branch into segment_length_um windows
    with overlap_um overlap (stride = segment_length_um - overlap_um). Score each window.
    Branch ordering uses PCA main-axis projection (fast, no graph traversal).
    """
    import time

    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def _tick(name: str) -> None:
        timings[name] = time.perf_counter() - t0

    zyx, x_um, y_um, z_um = load_flim_zyx(flim_path)
    _tick("load_flim")
    _, dendrite_mask, protrusion, skeleton_3d, _, _, _ = build_dendrite_and_protrusion_maps(
        zyx,
        x_um,
        dendrite_percentile=dendrite_percentile,
        soma_percentile=soma_percentile,
        threshold_area_um2=threshold_area_um2,
        shaft_opening_um=shaft_opening_um,
        min_skeleton_length_um=min_branch_length_um,
    )
    _tick("build_maps_skeleton")
    width_um_vol = dendrite_width_transform(dendrite_mask, x_um)
    _tick("width_edt_once")
    labeled_sk, n_branch = label_skeleton_branches(skeleton_3d)
    _tick("label_branches")

    min_vox_len = max(8, int(round(min_branch_length_um / x_um * 0.7)))
    min_len_loose = min_branch_length_um * 0.75

    sk_z, sk_y, sk_x = np.where(labeled_sk > 0)
    sk_labels = labeled_sk[sk_z, sk_y, sk_x]
    order = np.argsort(sk_labels, kind="stable")
    sk_z = sk_z[order]
    sk_y = sk_y[order]
    sk_x = sk_x[order]
    sk_labels = sk_labels[order]
    boundaries = np.searchsorted(sk_labels, np.arange(1, n_branch + 2))

    qualifying: list[dict[str, Any]] = []
    for branch_id in range(1, n_branch + 1):
        i0, i1 = int(boundaries[branch_id - 1]), int(boundaries[branch_id])
        n_pts = i1 - i0
        if n_pts < min_vox_len:
            continue
        path_est_um = n_pts * x_um * 0.85
        if path_est_um < min_len_loose:
            continue
        z = sk_z[i0:i1]
        y = sk_y[i0:i1]
        x = sk_x[i0:i1]
        widths_um = width_um_vol[z, y, x].astype(np.float64)
        mean_w = float(widths_um.mean())
        if mean_w <= min_branch_width_um or mean_w > max_branch_width_um:
            continue
        branch_pts = np.stack([z, y, x], axis=1).astype(np.int64)
        rep_zyx = branch_pts[n_pts // 2].astype(int)
        qualifying.append(
            {
                "branch_id": branch_id,
                "branch_pts": branch_pts,
                "widths_um": widths_um,
                "path_length_um": path_est_um,
                "rep_zyx": rep_zyx,
            }
        )
    _tick("filter_qualifying")

    if verbose:
        print(f"  qualifying branches (width/length): {len(qualifying)}")

    selected = select_spaced_branches(
        qualifying, max_branches, min_spacing_um, x_um, z_um
    )
    _tick("select_spaced")

    if verbose:
        print(f"  analyzing branches: {len(selected)} (max {max_branches})")

    puncta_zyx = _detect_puncta_near_branch_pts(
        protrusion, dendrite_mask, x_um, [s["branch_pts"] for s in selected]
    )
    _tick("detect_puncta_local")

    segment_rows: list[dict[str, Any]] = []
    branch_summaries: list[dict[str, Any]] = []
    spacing = np.array([z_um, x_um, x_um], dtype=np.float64)
    stride_um = max(1.0, segment_length_um - overlap_um)

    for info in selected:
        branch_id = info["branch_id"]
        branch_pts = info["branch_pts"]
        widths_um = info["widths_um"]
        path_len_um = info["path_length_um"]

        # order branch points along main axis via PCA projection (fast, no graph)
        b_phys = branch_pts.astype(np.float64) * spacing
        center = b_phys.mean(axis=0)
        bc = b_phys - center
        # first PC = direction of max variance
        cov = bc.T @ bc
        _, vecs = np.linalg.eigh(cov)
        main_axis = vecs[:, -1]
        proj = bc @ main_axis  # 1D arc-like coordinate
        sort_idx = np.argsort(proj)
        ordered_pts = branch_pts[sort_idx]
        ordered_proj = proj[sort_idx]
        # rescale proj to cumulative arc-like distance in µm
        arc_um = ordered_proj - ordered_proj[0]

        total_um = float(arc_um[-1]) if len(arc_um) > 1 else 0.0
        if total_um < 1.0:
            total_um = 1.0

        branch_seg_scores: list[float] = []
        seg_idx = 0
        win_start = 0.0
        while win_start < total_um - 1.0:
            win_end = win_start + segment_length_um
            mask = (arc_um >= win_start) & (arc_um < win_end)
            seg_pts = ordered_pts[mask]
            if len(seg_pts) < 3:
                win_start += stride_um
                continue

            scored = score_branch_point_set(
                seg_pts,
                dendrite_mask,
                protrusion,
                puncta_zyx,
                x_um,
                z_um,
                tube_radius_um=tube_radius_um,
                width_um_vol=width_um_vol,
            )
            if scored is None:
                win_start += stride_um
                continue

            mid_idx = len(seg_pts) // 2
            mid_pt = seg_pts[mid_idx].astype(int)
            row = {
                "branch_id": branch_id,
                "segment_index": seg_idx,
                "arc_start_um": float(win_start),
                "arc_end_um": float(min(win_end, total_um)),
                "path_length_um": path_len_um,
                "branch_mean_width_um": float(widths_um.mean()),
                "z_pix": int(mid_pt[0]),
                "y_pix": int(mid_pt[1]),
                "x_pix": int(mid_pt[2]),
                **scored,
            }
            segment_rows.append(row)
            branch_seg_scores.append(scored["segment_score"])
            seg_idx += 1
            win_start += stride_um

        if branch_seg_scores:
            branch_summaries.append(
                {
                    "branch_id": branch_id,
                    "path_length_um": path_len_um,
                    "branch_mean_width_um": float(widths_um.mean()),
                    "n_segments": len(branch_seg_scores),
                    "mean_segment_score": float(np.mean(branch_seg_scores)),
                    "max_segment_score": float(np.max(branch_seg_scores)),
                }
            )

    _tick("score_segments")

    segment_rows.sort(key=lambda r: r["segment_score"], reverse=True)
    branch_summaries.sort(key=lambda b: b["mean_segment_score"], reverse=True)

    return {
        "flim_path": flim_path,
        "zyx": zyx,
        "x_um": x_um,
        "y_um": y_um,
        "z_um": z_um,
        "skeleton_3d": skeleton_3d,
        "protrusion": protrusion,
        "dendrite_mask": dendrite_mask,
        "puncta_zyx": puncta_zyx,
        "segment_rows": segment_rows,
        "branch_summaries": branch_summaries,
        "n_branches": n_branch,
        "n_qualifying": len(qualifying),
        "n_analyzed": len(selected),
        "min_branch_width_um": min_branch_width_um,
        "max_branch_width_um": max_branch_width_um,
        "min_branch_length_um": min_branch_length_um,
        "segment_length_um": segment_length_um,
        "max_branches": max_branches,
        "timings_sec": timings,
    }
