# -*- coding: utf-8 -*-
"""
Find spiny (jagged) dendritic branches on low-mag .flim stacks without DeepD3.

Spiny branches look "fuzzy" on Z-MIP: many spine heads add local protrusions on the shaft.
We score each skeleton branch by:
  - protrusion density (white top-hat residual along the branch)
  - local puncta count (bright bumps near the branch)
  - boundary roughness on Z-MIP (perimeter^2 / area — jagged outline)
  - penalty for very thick segments (smooth main shaft)

Outputs same CSV layout as manual low-mag processing.
"""
import csv
import datetime
import glob
import os
import sys
from pathlib import Path

_CONTROLFLIMAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CONTROLFLIMAGE_DIR in sys.path:
    sys.path.remove(_CONTROLFLIMAGE_DIR)
sys.path.insert(0, _CONTROLFLIMAGE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    grey_opening,
    label as ndimage_label,
    median_filter,
)
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import perimeter_crofton, regionprops
from skimage.morphology import remove_small_objects

from after_click_image_func import (
    get_abs_um_pos_from_center_3d,
    save_image_with_assigned_pos_3d,
    save_pix_pos_from_click_list,
    save_um_pos_from_click_list,
)
from FLIMageAlignment import get_xyz_pixel_um
from FLIMageFileReader2 import FileReader
from utility.dendritic_shaft_detection import (
    define_skeleton_points,
    skeletonize_with_branch_filtering,
)


def export_path_from_flim(flim_path):
    return os.path.join(Path(flim_path).parent, Path(flim_path).stem)


def build_dendrite_and_protrusion_maps(
    zyx_array,
    xy_um,
    dendrite_percentile=94.0,
    soma_percentile=99.0,
    threshold_area_um2=10.0,
    shaft_opening_um=1.2,
    min_skeleton_length_um=None,
):
    """
    Build dendrite mask, soma-filtered skeleton, and spine protrusion (top-hat) map.
    """
    denoised = median_filter(zyx_array.astype(np.float32), size=(1, 3, 3))
    dendrite_thresh = np.percentile(denoised, dendrite_percentile)
    dendrite_mask = denoised > dendrite_thresh

    # Soma removal (same logic as generate_skeleton_from_filtered_image)
    from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure

    soma_thresh = np.percentile(denoised, soma_percentile)
    structure = generate_binary_structure(rank=3, connectivity=1)
    closed_soma = binary_closing(
        binary_opening(denoised > soma_thresh, structure=structure), structure=structure
    )
    labeled_soma, n_soma = ndimage_label(closed_soma, structure=structure)
    min_voxel_count = threshold_area_um2 / (xy_um * xy_um)
    soma_mask = np.zeros_like(closed_soma, dtype=bool)
    for i in range(1, n_soma + 1):
        if (labeled_soma == i).sum() >= min_voxel_count:
            soma_mask |= labeled_soma == i
    soma_mask = binary_dilation(soma_mask, structure=np.ones((3, 21, 21)))

    open_r = max(1, int(round(shaft_opening_um / xy_um)))
    footprint = np.ones((1, 2 * open_r + 1, 2 * open_r + 1))
    opened = grey_opening(denoised, footprint=footprint)
    protrusion = np.clip(denoised - opened, 0, None)
    protrusion[~dendrite_mask] = 0

    min_branch_pix = 8
    want_removed = min_skeleton_length_um is not None and min_skeleton_length_um > 0
    if want_removed:
        min_branch_pix = max(8, int(round(min_skeleton_length_um / xy_um)))

    sk_out = skeletonize_with_branch_filtering(
        dendrite_mask,
        min_branch_length=min_branch_pix,
        min_component_size=8,
        connection_radius=2,
        return_removed=want_removed,
    )
    skeleton_pruned_off = None
    skeleton_before_length_prune = None
    if want_removed:
        skeleton_3d, _, skeleton_pruned_off, skeleton_before_length_prune = sk_out
    else:
        skeleton_3d, _ = sk_out

    def _apply_soma_mask(sk: np.ndarray) -> np.ndarray:
        coords = np.array(np.where(sk)).T
        if len(coords) == 0:
            return sk
        out = np.zeros_like(sk, dtype=bool)
        valid = ~soma_mask[tuple(coords.T)]
        coords = coords[valid]
        if len(coords) > 0:
            out[tuple(coords.T)] = True
        return out

    skeleton_3d = _apply_soma_mask(skeleton_3d)
    if skeleton_pruned_off is not None:
        skeleton_pruned_off = _apply_soma_mask(skeleton_pruned_off)
    if skeleton_before_length_prune is not None:
        skeleton_before_length_prune = _apply_soma_mask(skeleton_before_length_prune)

    return (
        denoised,
        dendrite_mask,
        protrusion,
        skeleton_3d,
        soma_mask,
        skeleton_pruned_off,
        skeleton_before_length_prune,
    )


def detect_puncta_near_dendrite(
    protrusion,
    dendrite_mask,
    xy_um,
    min_distance_um=0.8,
    threshold_percentile=70.0,
):
    """Local maxima on protrusion map (spine-head bumps), per Z slice."""
    min_dist_pix = max(1, int(round(min_distance_um / xy_um)))
    pos = protrusion[protrusion > 0]
    if len(pos) == 0:
        return np.zeros((0, 3), dtype=int)
    thresh = np.percentile(pos, threshold_percentile)

    puncta = []
    for z in range(protrusion.shape[0]):
        sl = protrusion[z]
        if sl.max() <= 0:
            continue
        peaks = peak_local_max(
            sl,
            min_distance=min_dist_pix,
            threshold_abs=thresh,
            exclude_border=True,
        )
        for y, x in peaks:
            if dendrite_mask[z, y, x]:
                puncta.append([z, int(y), int(x)])
    return np.array(puncta, dtype=int) if puncta else np.zeros((0, 3), dtype=int)


def label_skeleton_branches(skeleton_3d):
    labeled, n = ndimage_label(skeleton_3d, structure=np.ones((3, 3, 3)))
    return labeled, n


def branch_corridor_mask(labeled_skeleton, branch_id, xy_um, z_um, tube_radius_um):
    branch_pts = np.array(np.where(labeled_skeleton == branch_id)).T
    if len(branch_pts) == 0:
        return np.zeros_like(labeled_skeleton, dtype=bool), branch_pts
    vol = np.zeros_like(labeled_skeleton, dtype=bool)
    vol[tuple(branch_pts.T)] = True
    ry = max(1, int(round(tube_radius_um / xy_um)))
    rz = max(1, int(round(tube_radius_um / z_um)))
    dilated = binary_dilation(vol, structure=np.ones((2 * rz + 1, 2 * ry + 1, 2 * ry + 1)))
    return dilated, branch_pts


def mip_branch_roughness(dendrite_mip, corridor_mip):
    """Higher = more jagged outline (spiny branch on Z-MIP)."""
    region = dendrite_mip & corridor_mip
    region = remove_small_objects(region, min_size=16)
    if not region.any():
        return 0.0
    labeled, n = ndimage_label(region)
    if n == 0:
        return 0.0
    props = regionprops(labeled)[0]
    if props.area < 1:
        return 0.0
    perim = perimeter_crofton(labeled == props.label)
    return float(perim ** 2 / props.area)


def score_skeleton_branch(
    branch_id,
    labeled_skeleton,
    protrusion,
    dendrite_mask,
    puncta_zyx,
    xy_um,
    z_um,
    tube_radius_um=3.5,
    max_shaft_thickness_um=3.0,
):
    """
    Combined spiny-branch score (unitless, higher = more spiny/jagged).
    """
    corridor, branch_pts = branch_corridor_mask(
        labeled_skeleton, branch_id, xy_um, z_um, tube_radius_um
    )
    if len(branch_pts) < 5:
        return None

    sk_len = len(branch_pts)
    prot_vals = protrusion[corridor]
    protrusion_density = float(prot_vals.sum() / sk_len)
    protrusion_mean = float(prot_vals.mean()) if prot_vals.size else 0.0

    dt = distance_transform_edt(dendrite_mask)
    mean_thickness_um = float(dt[corridor].mean() * xy_um)
    if mean_thickness_um > max_shaft_thickness_um:
        thickness_penalty = 0.25
    else:
        thickness_penalty = 1.0

    if len(puncta_zyx) > 0:
        p_phys = puncta_zyx * np.array([z_um, xy_um, xy_um])
        b_phys = branch_pts * np.array([z_um, xy_um, xy_um])
        tree = cKDTree(b_phys)
        dists, _ = tree.query(p_phys, k=1)
        puncta_count = int(np.sum(dists <= tube_radius_um))
    else:
        puncta_count = 0

    dendrite_mip = dendrite_mask.max(axis=0)
    corridor_mip = corridor.max(axis=0)
    roughness = mip_branch_roughness(dendrite_mip, corridor_mip)

    spiny_score = thickness_penalty * (
        protrusion_density
        + 2.0 * puncta_count / sk_len
        + 0.15 * roughness
        + 0.5 * protrusion_mean
    )

    return {
        "branch_id": branch_id,
        "spiny_score": spiny_score,
        "protrusion_density": protrusion_density,
        "puncta_count": puncta_count,
        "roughness": roughness,
        "mean_thickness_um": mean_thickness_um,
        "skeleton_length_pix": sk_len,
        "branch_pts": branch_pts,
    }


def pick_best_point_on_branch(branch_pts, protrusion, puncta_zyx, xy_um, z_um, local_radius_um):
    ry = max(1, int(round(local_radius_um / xy_um)))
    rz = max(1, int(round(local_radius_um / z_um)))
    best_pt = branch_pts[len(branch_pts) // 2]
    best_local = -1.0
    for pt in branch_pts:
        z0, y0, x0 = int(pt[0]), int(pt[1]), int(pt[2])
        zsl = slice(max(0, z0 - rz), min(protrusion.shape[0], z0 + rz + 1))
        ysl = slice(max(0, y0 - ry), min(protrusion.shape[1], y0 + ry + 1))
        xsl = slice(max(0, x0 - ry), min(protrusion.shape[2], x0 + ry + 1))
        local_prot = float(protrusion[zsl, ysl, xsl].sum())
        local_puncta = 0
        if len(puncta_zyx) > 0:
            dz = np.abs(puncta_zyx[:, 0] - z0)
            dy = puncta_zyx[:, 1] - y0
            dx = puncta_zyx[:, 2] - x0
            local_puncta = int(
                np.sum((dz <= rz) & (dy * dy + dx * dx <= ry * ry))
            )
        local_score = local_prot + 3.0 * local_puncta
        if local_score > best_local:
            best_local = local_score
            best_pt = pt
    return best_pt.astype(int), best_local


def physical_distance_um(p1_zyx, p2_zyx, xy_um, z_um):
    dz = (p1_zyx[0] - p2_zyx[0]) * z_um
    dy = (p1_zyx[1] - p2_zyx[1]) * xy_um
    dx = (p1_zyx[2] - p2_zyx[2]) * xy_um
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def select_spaced_positions(candidates, max_num, min_spacing_um, xy_um, z_um):
    ordered = sorted(candidates, key=lambda c: c["spiny_score"], reverse=True)
    selected = []
    for cand in ordered:
        if len(selected) >= max_num:
            break
        if all(
            physical_distance_um(cand["zyx"], s["zyx"], xy_um, z_um) >= min_spacing_um
            for s in selected
        ):
            selected.append(cand)
    return selected


def fallback_skeleton_positions(skeleton_3d, spacing_um, x_um, y_um, z_um, max_num):
    points = define_skeleton_points(skeleton_3d, spacing_um, x_um, y_um, z_um)
    if len(points) == 0:
        return []
    sorted_pts = points[points[:, 0].argsort()[::-1]][:max_num]
    return [
        {
            "branch_id": -1,
            "zyx": pt.astype(int),
            "spiny_score": 0.0,
            "puncta_count": 0,
            "roughness": 0.0,
        }
        for pt in sorted_pts
    ]


def save_branch_overview_png(
    zyx_array,
    skeleton_3d,
    protrusion,
    puncta_zyx,
    selected,
    savepath,
):
    mip = zyx_array.max(axis=0)
    sk_mip = skeleton_3d.max(axis=0)
    prot_mip = protrusion.max(axis=0)
    vmax = np.percentile(mip, 99.5) if mip.max() > 0 else 1
    pmax = np.percentile(prot_mip[prot_mip > 0], 95) if (prot_mip > 0).any() else 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.imshow(mip, cmap="gray", vmax=vmax)
    ax.imshow(np.ma.masked_where(prot_mip <= 0, prot_mip), cmap="magma", alpha=0.45, vmax=pmax)
    ax.imshow(np.ma.masked_where(~sk_mip, sk_mip), cmap="Greens", alpha=0.25)
    if len(puncta_zyx) > 0:
        ax.scatter(puncta_zyx[:, 2], puncta_zyx[:, 1], s=6, c="cyan", alpha=0.6)
    for cand in selected:
        x, y = cand["zyx"][2], cand["zyx"][1]
        ax.scatter(x, y, s=100, facecolors="none", edgecolors="yellow", linewidths=2)
        ax.text(
            x + 2,
            y + 2,
            f"b{cand['branch_id']}: {cand['spiny_score']:.2f}",
            color="yellow",
            fontsize=7,
        )
    ax.set_title("Z MIP: protrusion (warm) + picks (yellow)")
    ax.axis("off")

    axes[1].imshow(prot_mip, cmap="hot", vmax=pmax)
    axes[1].set_title("Protrusion map (top-hat, Z-MIP)")
    axes[1].axis("off")
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_branch_scores_csv(candidates, selected, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "branch_id",
                "z_pix",
                "y_pix",
                "x_pix",
                "spiny_score",
                "protrusion_density",
                "puncta_count",
                "roughness",
                "mean_thickness_um",
                "selected",
            ],
        )
        writer.writeheader()
        selected_set = {tuple(c["zyx"]) for c in selected}
        for cand in candidates:
            z, y, x = cand["zyx"]
            writer.writerow(
                {
                    "branch_id": cand["branch_id"],
                    "z_pix": int(z),
                    "y_pix": int(y),
                    "x_pix": int(x),
                    "spiny_score": cand.get("spiny_score", 0),
                    "protrusion_density": cand.get("protrusion_density", 0),
                    "puncta_count": cand.get("puncta_count", 0),
                    "roughness": cand.get("roughness", 0),
                    "mean_thickness_um": cand.get("mean_thickness_um", 0),
                    "selected": int(tuple(cand["zyx"]) in selected_set),
                }
            )


def find_and_save_spiny_branches(
    flim_path,
    max_pos_cand_num=8,
    min_spacing_um=20.0,
    local_radius_um=10.0,
    dendrite_percentile=94.0,
    soma_percentile=99.0,
    threshold_area_um2=10.0,
    shaft_opening_um=1.2,
    tube_radius_um=3.5,
    max_shaft_thickness_um=3.0,
    min_branch_score=0.0,
    min_skeleton_length_pix=12,
    skip_if_defined=False,
):
    export_path = export_path_from_flim(flim_path)
    os.makedirs(export_path, exist_ok=True)
    pos_pix_csv = os.path.join(export_path, "assigned_pixel_pos.csv")
    pos_um_csv = os.path.join(export_path, "assigned_relative_um_pos.csv")

    if skip_if_defined and os.path.exists(pos_pix_csv):
        print(f"Skipping (already defined): {flim_path}")
        return []

    print(f"\n=== low-mag spiny branch finder: {flim_path} ===")
    iminfo = FileReader()
    iminfo.read_imageFile(flim_path, True)
    zyx_array = np.array(iminfo.image).sum(axis=tuple([1, 2, 5]))
    x_um, y_um, z_um = get_xyz_pixel_um(iminfo)
    print(
        f"  stack Z,Y,X = {zyx_array.shape}, pixel um xy={x_um:.3f} z={z_um:.3f}"
    )

    denoised, dendrite_mask, protrusion, skeleton_3d, _, _, _ = build_dendrite_and_protrusion_maps(
        zyx_array,
        x_um,
        dendrite_percentile=dendrite_percentile,
        soma_percentile=soma_percentile,
        threshold_area_um2=threshold_area_um2,
        shaft_opening_um=shaft_opening_um,
    )
    puncta_zyx = detect_puncta_near_dendrite(protrusion, dendrite_mask, x_um)
    print(f"  puncta (local max on protrusion): {len(puncta_zyx)}")

    labeled_sk, n_branch = label_skeleton_branches(skeleton_3d)
    print(f"  skeleton branches: {n_branch}")

    branch_candidates = []
    for branch_id in range(1, n_branch + 1):
        scored = score_skeleton_branch(
            branch_id,
            labeled_sk,
            protrusion,
            dendrite_mask,
            puncta_zyx,
            x_um,
            z_um,
            tube_radius_um=tube_radius_um,
            max_shaft_thickness_um=max_shaft_thickness_um,
        )
        if scored is None:
            continue
        if scored["skeleton_length_pix"] < min_skeleton_length_pix:
            continue
        if scored["spiny_score"] < min_branch_score:
            continue
        zyx_pt, _ = pick_best_point_on_branch(
            scored["branch_pts"],
            protrusion,
            puncta_zyx,
            x_um,
            z_um,
            local_radius_um,
        )
        branch_candidates.append(
            {
                "branch_id": branch_id,
                "zyx": zyx_pt,
                "spiny_score": scored["spiny_score"],
                "protrusion_density": scored["protrusion_density"],
                "puncta_count": scored["puncta_count"],
                "roughness": scored["roughness"],
                "mean_thickness_um": scored["mean_thickness_um"],
            }
        )

    branch_candidates.sort(key=lambda c: c["spiny_score"], reverse=True)
    print(f"  branches passing score filter: {len(branch_candidates)}")

    if branch_candidates:
        selected = select_spaced_positions(
            branch_candidates, max_pos_cand_num, min_spacing_um, x_um, z_um
        )
    else:
        print("  no spiny branches — fallback to skeleton spacing")
        selected = fallback_skeleton_positions(
            skeleton_3d, min_spacing_um, x_um, y_um, z_um, max_pos_cand_num
        )

    selected_zyx = [c["zyx"].tolist() for c in selected]
    save_pix_pos_from_click_list(selected_zyx, csv_savepath=pos_pix_csv)
    zyx_um_dict = get_abs_um_pos_from_center_3d(iminfo.statedict, selected_zyx)
    save_um_pos_from_click_list(zyx_um_dict, csv_savepath=pos_um_csv)

    overview_path = os.path.join(export_path, "spiny_branch_overview.png")
    save_branch_overview_png(
        zyx_array, skeleton_3d, protrusion, puncta_zyx, selected, overview_path
    )
    scores_path = os.path.join(export_path, "spiny_branch_scores.csv")
    save_branch_scores_csv(branch_candidates, selected, scores_path)

    prot_path = os.path.join(export_path, "protrusion_z_mip.png")
    plt.imsave(prot_path, protrusion.max(axis=0), cmap="hot")

    save_image_with_assigned_pos_3d(
        tif_path="",
        pix_pos_csv_path=pos_pix_csv,
        png_savefolder=export_path,
        input_arr=True,
        array=zyx_array,
    )

    print(f"  saved {len(selected_zyx)} positions -> {pos_pix_csv}")
    for cand in selected:
        z, y, x = cand["zyx"]
        print(
            f"    branch {cand['branch_id']}: zyx=({z},{y},{x}), "
            f"spiny_score={cand['spiny_score']:.3f}, "
            f"puncta={cand['puncta_count']}, "
            f"thick={cand.get('mean_thickness_um', 0):.2f} um"
        )
    return selected_zyx


# Backward-compatible alias
find_and_save_mushroom_rich_branches = find_and_save_spiny_branches


def filter_by_modification_time(flim_list, min_seconds_since_modification=20):
    filtered = []
    for flim_path in flim_list:
        delta = (
            datetime.datetime.now()
            - datetime.datetime.fromtimestamp(os.path.getmtime(flim_path))
        ).total_seconds()
        if delta > min_seconds_since_modification:
            filtered.append(flim_path)
    return filtered


def run_lowmag_mushroom_branch_finder(
    savefolder,
    filename_pattern="*_001.flim",
    max_pos_cand_num=8,
    min_spacing_um=20.0,
    local_radius_um=10.0,
    dendrite_percentile=94.0,
    shaft_opening_um=1.2,
    tube_radius_um=3.5,
    max_shaft_thickness_um=3.0,
    min_branch_score=0.0,
    min_seconds_since_modification=20,
    skip_if_defined=True,
    **kwargs,
):
    """Process all matching low-mag .flim files (texture-based, no DeepD3)."""
    flim_list = glob.glob(os.path.join(savefolder, filename_pattern))
    flim_list = filter_by_modification_time(flim_list, min_seconds_since_modification)
    print(f"Found {len(flim_list)} low-mag file(s) matching {filename_pattern}")
    print(
        "Mode: spiny-branch texture (top-hat protrusion + puncta + boundary roughness)"
    )

    results = {}
    for flim_path in flim_list:
        try:
            selected = find_and_save_spiny_branches(
                flim_path,
                max_pos_cand_num=max_pos_cand_num,
                min_spacing_um=min_spacing_um,
                local_radius_um=local_radius_um,
                dendrite_percentile=dendrite_percentile,
                shaft_opening_um=shaft_opening_um,
                tube_radius_um=tube_radius_um,
                max_shaft_thickness_um=max_shaft_thickness_um,
                min_branch_score=min_branch_score,
                skip_if_defined=skip_if_defined,
            )
            results[flim_path] = selected
        except Exception as exc:
            print(f"  ERROR on {flim_path}: {exc}")
            import traceback

            traceback.print_exc()
    return results
