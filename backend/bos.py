"""
Background-Oriented Schlieren (BOS) core pipeline.

For each pair of (previous, current) frames we:
  1. Estimate the camera's background motion between the two frames.
     Two interchangeable strategies are supported:
       - "features": detect keypoints (ORB / AKAZE / SIFT), match descriptors,
         then find a homography with RANSAC (moving objects fall out as outliers).
       - "optical_flow": track sparse points with calcOpticalFlowPyrLK, then find
         a homography with RANSAC on the tracked point pairs.
  2. Warp (align) the current frame back onto the previous frame using the
     homography H, cancelling out the global camera shift.
  3. Compute cv2.absdiff between the previous frame and the aligned current frame.
     What remains is mostly the density-gradient "shimmer" (schlieren) plus any
     residual moving objects.
  4. Post-process the difference to amplify / highlight the schlieren effect
     (blur, gain, threshold, morphology, normalization, colormap, overlay).

The module is deliberately dependency-light (cv2 + numpy only) so it runs inside
the free-tier memory budget and has no external services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# ── Parameters ──────────────────────────────────────────────────────────────

# AKAZE is absent from some OpenCV builds (e.g. headless 5.x wheels), so we
# only advertise detectors that are reliably present.
_FEATURE_DETECTORS = ("orb", "sift")
_COLORMAPS: dict[str, int] = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
}


@dataclass
class BOSParams:
    """Tunable knobs for the BOS pipeline. All have sensible defaults so the
    endpoint works with an empty request body."""

    # ── Motion-estimation strategy ──────────────────────────────────────────
    method: str = "features"                # "features" | "optical_flow"

    # Feature-based (method == "features")
    detector: str = "orb"                   # "orb" | "akaze" | "sift"
    max_features: int = 2000                # keypoint budget (orb/sift)
    match_ratio: float = 0.75               # Lowe ratio test threshold
    min_matches: int = 12                   # need at least this many good matches

    # Optical-flow based (method == "optical_flow")
    flow_max_corners: int = 600             # goodFeaturesToTrack maxCorners
    flow_quality: float = 0.01              # goodFeaturesToTrack qualityLevel
    flow_min_distance: float = 7.0          # goodFeaturesToTrack minDistance
    flow_win_size: int = 21                 # LK window size

    # Homography / RANSAC
    ransac_thresh: float = 4.0              # RANSAC reprojection threshold (px)

    # ── Schlieren post-processing ───────────────────────────────────────────
    blur_ksize: int = 5                     # Gaussian blur kernel (odd, 0 disables)
    gain: float = 4.0                       # brightness multiplier on the diff
    threshold: int = 8                      # drop diff values below this (0 disables)
    morph_ksize: int = 3                    # morphological open kernel (0 disables)
    normalize: bool = True                  # stretch contrast to full 0-255 range
    colormap: str = "inferno"              # see _COLORMAPS, or "none"/"gray"
    overlay: float = 0.0                    # 0 = pure schlieren; >0 blends over frame
    border_erode: int = 5                   # erode the valid-warp mask by this (px)

    def normalized(self) -> "BOSParams":
        """Return a validated copy with values coerced to safe ranges."""
        p = BOSParams(**{f.name: getattr(self, f.name) for f in _fields(self)})
        p.method = p.method if p.method in ("features", "optical_flow") else "features"
        p.detector = p.detector if p.detector in _FEATURE_DETECTORS else "orb"
        p.max_features = int(_clamp(p.max_features, 100, 20000))
        p.match_ratio = float(_clamp(p.match_ratio, 0.1, 0.99))
        p.min_matches = int(_clamp(p.min_matches, 4, 5000))
        p.flow_max_corners = int(_clamp(p.flow_max_corners, 20, 5000))
        p.flow_quality = float(_clamp(p.flow_quality, 0.0001, 0.5))
        p.flow_min_distance = float(_clamp(p.flow_min_distance, 1.0, 100.0))
        p.flow_win_size = int(_clamp(_odd(p.flow_win_size), 3, 101))
        p.ransac_thresh = float(_clamp(p.ransac_thresh, 0.5, 50.0))
        p.blur_ksize = 0 if p.blur_ksize <= 0 else int(_clamp(_odd(p.blur_ksize), 3, 51))
        p.gain = float(_clamp(p.gain, 0.1, 50.0))
        p.threshold = int(_clamp(p.threshold, 0, 255))
        p.morph_ksize = 0 if p.morph_ksize <= 0 else int(_clamp(p.morph_ksize, 1, 51))
        p.colormap = p.colormap if p.colormap in _COLORMAPS or p.colormap in ("none", "gray") else "inferno"
        p.overlay = float(_clamp(p.overlay, 0.0, 1.0))
        p.border_erode = int(_clamp(p.border_erode, 0, 100))
        return p


def _fields(obj: Any):
    from dataclasses import fields as _f
    return _f(obj)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _odd(v: int) -> int:
    v = int(v)
    return v if v % 2 == 1 else v + 1


# ── Step 1: motion estimation ────────────────────────────────────────────────

def _build_detector(params: BOSParams):
    d = params.detector
    if d == "sift" and hasattr(cv2, "SIFT_create"):
        # SIFT is patent-free and bundled with modern OpenCV.
        return cv2.SIFT_create(nfeatures=params.max_features)
    return cv2.ORB_create(nfeatures=params.max_features)


def estimate_homography_features(
    prev_gray: np.ndarray, curr_gray: np.ndarray, params: BOSParams
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Detect + match keypoints, then RANSAC homography mapping curr -> prev."""
    stats: dict[str, Any] = {"method": "features", "detector": params.detector}
    detector = _build_detector(params)
    kp1, des1 = detector.detectAndCompute(prev_gray, None)
    kp2, des2 = detector.detectAndCompute(curr_gray, None)
    stats["keypoints_prev"] = len(kp1) if kp1 else 0
    stats["keypoints_curr"] = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or len(kp1) < params.min_matches or len(kp2) < params.min_matches:
        stats["good_matches"] = 0
        stats["inliers"] = 0
        return None, stats

    # Float descriptors (SIFT) use L2; binary descriptors (ORB) use Hamming.
    norm = cv2.NORM_L2 if np.issubdtype(des1.dtype, np.floating) else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < params.match_ratio * n.distance:
            good.append(m)
    stats["good_matches"] = len(good)

    if len(good) < params.min_matches:
        stats["inliers"] = 0
        return None, stats

    # queryIdx -> prev (dst), trainIdx -> curr (src). We map curr onto prev.
    src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, params.ransac_thresh)
    stats["inliers"] = int(mask.sum()) if mask is not None else 0
    return H, stats


def estimate_homography_flow(
    prev_gray: np.ndarray, curr_gray: np.ndarray, params: BOSParams
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Track sparse corners with pyramidal Lucas-Kanade, then RANSAC homography."""
    stats: dict[str, Any] = {"method": "optical_flow"}
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=params.flow_max_corners,
        qualityLevel=params.flow_quality,
        minDistance=params.flow_min_distance,
        blockSize=7,
    )
    stats["tracked_points"] = 0
    stats["good_matches"] = 0
    stats["inliers"] = 0
    if p0 is None or len(p0) < params.min_matches:
        stats["keypoints_prev"] = 0 if p0 is None else len(p0)
        return None, stats
    stats["keypoints_prev"] = len(p0)

    lk_params = dict(
        winSize=(params.flow_win_size, params.flow_win_size),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    if p1 is None or st is None:
        return None, stats

    st = st.reshape(-1).astype(bool)
    good_prev = p0.reshape(-1, 2)[st]
    good_curr = p1.reshape(-1, 2)[st]
    stats["tracked_points"] = int(st.sum())
    stats["good_matches"] = int(st.sum())

    if len(good_prev) < params.min_matches:
        return None, stats

    # Map curr -> prev.
    H, mask = cv2.findHomography(good_curr, good_prev, cv2.RANSAC, params.ransac_thresh)
    stats["inliers"] = int(mask.sum()) if mask is not None else 0
    return H, stats


# ── Step 2/3: warp + difference ──────────────────────────────────────────────

def _valid_mask(shape: tuple[int, int], H: np.ndarray, erode: int) -> np.ndarray:
    """Mask (255 where valid) of pixels covered by the warped current frame,
    so black border regions from the warp don't create fake schlieren signal."""
    h, w = shape
    ones = np.full((h, w), 255, np.uint8)
    warped = cv2.warpPerspective(ones, H, (w, h))
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1))
        warped = cv2.erode(warped, k)
    return warped


# ── Step 4: schlieren post-processing ────────────────────────────────────────

def postprocess(diff: np.ndarray, params: BOSParams) -> np.ndarray:
    """Amplify and colorize the raw absdiff into a schlieren visualization (BGR)."""
    out = diff
    if params.blur_ksize >= 3:
        out = cv2.GaussianBlur(out, (params.blur_ksize, params.blur_ksize), 0)
    if params.gain != 1.0:
        out = cv2.convertScaleAbs(out, alpha=params.gain, beta=0)
    if params.threshold > 0:
        _, out = cv2.threshold(out, params.threshold, 255, cv2.THRESH_TOZERO)
    if params.morph_ksize >= 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.morph_ksize, params.morph_ksize)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if params.normalize and out.max() > 0:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)

    if params.colormap in ("none", "gray"):
        vis = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    else:
        vis = cv2.applyColorMap(out, _COLORMAPS[params.colormap])
    return vis


def compute_schlieren(
    prev_bgr: np.ndarray, curr_bgr: np.ndarray, params: BOSParams
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the full 4-step BOS pipeline on one consecutive frame pair.

    Returns (visualization_bgr, stats)."""
    params = params.normalized()
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    # Step 1: estimate global background motion (homography).
    if params.method == "optical_flow":
        H, stats = estimate_homography_flow(prev_gray, curr_gray, params)
    else:
        H, stats = estimate_homography_features(prev_gray, curr_gray, params)

    h, w = prev_gray.shape
    # Step 2: warp current frame back onto previous (align away camera shift).
    if H is not None:
        aligned = cv2.warpPerspective(curr_gray, H, (w, h))
        valid = _valid_mask((h, w), H, params.border_erode)
        stats["aligned"] = True
    else:
        # No reliable motion model — fall back to raw (unaligned) difference.
        aligned = curr_gray
        valid = np.full((h, w), 255, np.uint8)
        stats["aligned"] = False

    # Step 3: absolute difference between prev and aligned current.
    diff = cv2.absdiff(prev_gray, aligned)
    diff = cv2.bitwise_and(diff, diff, mask=valid)

    # Step 4: post-process to highlight the schlieren shimmer.
    vis = postprocess(diff, params)

    if params.overlay > 0.0:
        vis = cv2.addWeighted(prev_bgr, 1.0 - params.overlay, vis, params.overlay, 0)

    stats["diff_mean"] = float(diff.mean())
    stats["diff_max"] = int(diff.max())
    return vis, stats
