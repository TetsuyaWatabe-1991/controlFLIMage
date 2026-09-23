"""Rigid-shift estimators that are not cross-correlation.

Optical flow and feature matching use the whole projection. Each returns
object displacement relative to frame 0, in (dy, dx) pixels, positive dy
toward larger row indices.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.registration import optical_flow_ilk, optical_flow_tvl1

def _to_uint8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _robust_translation(dy: np.ndarray, dx: np.ndarray) -> tuple[float, float]:
    """Median translation, after dropping vectors far from the first median."""
    dy = np.asarray(dy, dtype=np.float64).ravel()
    dx = np.asarray(dx, dtype=np.float64).ravel()
    ok = np.isfinite(dy) & np.isfinite(dx)
    dy, dx = dy[ok], dx[ok]
    if dy.size < 8:
        return 0.0, 0.0
    med_y = float(np.median(dy))
    med_x = float(np.median(dx))
    near = (dy - med_y) ** 2 + (dx - med_x) ** 2 <= 9.0
    if int(near.sum()) >= 8:
        med_y = float(np.median(dy[near]))
        med_x = float(np.median(dx[near]))
    return med_y, med_x


def _textured_samples(reference: np.ndarray, flow_y: np.ndarray, flow_x: np.ndarray) -> tuple[float, float]:
    """Median flow on pixels with a strong gradient, so blank background does not vote."""
    gy, gx = np.gradient(np.asarray(reference, dtype=np.float32))
    magnitude = np.hypot(gy, gx)
    threshold = float(np.percentile(magnitude, 75))
    keep = magnitude >= max(threshold, 1e-6)
    if int(keep.sum()) < 8:
        keep = np.ones(reference.shape, dtype=bool)
    return _robust_translation(flow_y[keep], flow_x[keep])


def farneback(stack: np.ndarray) -> np.ndarray:
    """Dense Farneback flow. The median motion of textured pixels is the shift."""
    shifts = np.zeros((len(stack), 2), dtype=np.float64)
    reference = _to_uint8(stack[0])
    for t in range(1, len(stack)):
        flow = cv2.calcOpticalFlowFarneback(
            reference,
            _to_uint8(stack[t]),
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        shifts[t] = _textured_samples(reference.astype(np.float32), flow[..., 1], flow[..., 0])
    return shifts


def pyramidal_lucas_kanade(stack: np.ndarray) -> np.ndarray:
    """Sparse pyramidal Lucas-Kanade on Shi-Tomasi corners."""
    shifts = np.zeros((len(stack), 2), dtype=np.float64)
    reference = _to_uint8(stack[0])
    corners = cv2.goodFeaturesToTrack(
        reference,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5,
    )
    if corners is None:
        return shifts
    for t in range(1, len(stack)):
        nxt, status, _err = cv2.calcOpticalFlowPyrLK(
            reference,
            _to_uint8(stack[t]),
            corners,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if nxt is None or status is None:
            continue
        good = status.ravel() == 1
        if int(good.sum()) < 8:
            continue
        delta = nxt[good] - corners[good]
        # OpenCV points are (x, y).
        shifts[t] = _robust_translation(delta[:, 0, 1], delta[:, 0, 0])
    return shifts


def iterative_lucas_kanade(stack: np.ndarray) -> np.ndarray:
    """Dense iterative Lucas-Kanade. The median flow is the object displacement."""
    shifts = np.zeros((len(stack), 2), dtype=np.float64)
    reference = np.asarray(stack[0], dtype=np.float32)
    for t in range(1, len(stack)):
        flow_y, flow_x = optical_flow_ilk(
            reference,
            np.asarray(stack[t], dtype=np.float32),
            radius=7,
            num_warp=4,
        )
        dy, dx = _textured_samples(reference, flow_y, flow_x)
        shifts[t] = (dy, dx)
    return shifts


def tvl1(stack: np.ndarray) -> np.ndarray:
    """TV-L1 optical flow, median over textured pixels."""
    shifts = np.zeros((len(stack), 2), dtype=np.float64)
    reference = np.asarray(stack[0], dtype=np.float32)
    for t in range(1, len(stack)):
        flow_y, flow_x = optical_flow_tvl1(
            reference,
            np.asarray(stack[t], dtype=np.float32),
            num_warp=3,
            num_iter=8,
        )
        dy, dx = _textured_samples(reference, flow_y, flow_x)
        shifts[t] = (dy, dx)
    return shifts


def orb_ransac(stack: np.ndarray) -> np.ndarray:
    """ORB matches, then a RANSAC translation. No rotation is fit."""
    shifts = np.zeros((len(stack), 2), dtype=np.float64)
    orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=6)
    reference = _to_uint8(stack[0])
    key_ref, desc_ref = orb.detectAndCompute(reference, None)
    if desc_ref is None or len(key_ref) < 8:
        return shifts
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    for t in range(1, len(stack)):
        key_mov, desc_mov = orb.detectAndCompute(_to_uint8(stack[t]), None)
        if desc_mov is None or len(key_mov) < 8:
            continue
        pairs = matcher.knnMatch(desc_ref, desc_mov, k=2)
        src = []
        dst = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < 0.75 * second.distance:
                src.append(key_ref[best.queryIdx].pt)
                dst.append(key_mov[best.trainIdx].pt)
        if len(src) < 8:
            continue
        src_xy = np.asarray(src, dtype=np.float32)
        dst_xy = np.asarray(dst, dtype=np.float32)
        affine, inliers = cv2.estimateAffinePartial2D(
            src_xy,
            dst_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )
        if affine is None or inliers is None or int(inliers.sum()) < 8:
            delta = dst_xy - src_xy
            shifts[t] = _robust_translation(delta[:, 1], delta[:, 0])
            continue
        # Partial affine is [a, -b, tx; b, a, ty]. Use the inlier translation only.
        kept = inliers.ravel() == 1
        delta = dst_xy[kept] - src_xy[kept]
        shifts[t] = _robust_translation(delta[:, 1], delta[:, 0])
    return shifts


OTHER_ALIGNERS = {
    "farneback": farneback,
    "pyramidal_lucas_kanade": pyramidal_lucas_kanade,
    "iterative_lucas_kanade": iterative_lucas_kanade,
    "tvl1": tvl1,
    "orb_ransac": orb_ransac,
}
