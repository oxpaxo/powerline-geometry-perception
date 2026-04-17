# projects/powerline_v1/utils/geometric_verifier.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ComponentStats:
    label: int
    area: int
    bbox: Tuple[int, int, int, int]        # x, y, w, h
    centroid: Tuple[float, float]          # x, y
    length: float
    width: float
    aspect_ratio: float
    angle_deg: float                       # [0, 180)
    center_mean: float
    orient_consistency: float              # abs(dot(local_orient, major_axis)) mean
    dist_mean: float                       # optional, np.nan if distance not provided
    dist_q80: float                        # optional, np.nan if distance not provided


def _ensure_float01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if x.max() > 1.0 or x.min() < 0.0:
        x = np.clip(x, 0.0, 1.0)
    return x


def _ensure_orient_hw2(orient_map: np.ndarray) -> np.ndarray:
    """
    Accept either [2, H, W] or [H, W, 2], return [H, W, 2].
    Channel convention:
        [..., 0] = ox
        [..., 1] = oy
    """
    orient_map = np.asarray(orient_map, dtype=np.float32)
    if orient_map.ndim != 3:
        raise ValueError(f'orient_map must be 3D, but got shape={orient_map.shape}')

    if orient_map.shape[0] == 2:
        orient_map = np.transpose(orient_map, (1, 2, 0))
    elif orient_map.shape[2] == 2:
        pass
    else:
        raise ValueError(
            f'orient_map must be [2,H,W] or [H,W,2], but got shape={orient_map.shape}'
        )
    return orient_map


def _normalize_orient(orient_map: np.ndarray) -> np.ndarray:
    orient_map = np.asarray(orient_map, dtype=np.float32)
    norm = np.linalg.norm(orient_map, axis=-1, keepdims=True) + 1e-6
    return orient_map / norm


def _angle_deg_from_vec(vec_xy: np.ndarray) -> float:
    angle = np.degrees(np.arctan2(float(vec_xy[1]), float(vec_xy[0])))
    angle = angle % 180.0  # direction sign is symmetric for line orientation
    return float(angle)


def _angle_diff_deg(a: float, b: float) -> float:
    """
    Line orientation distance in [0, 90].
    Since line direction is symmetric, 0° and 180° are equivalent.
    """
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def _principal_axis_from_mask(component_mask: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Args:
        component_mask: [H, W] bool/uint8, a single connected component

    Returns:
        major_unit_xy: np.ndarray shape [2]
        length: float
        width: float
        angle_deg: float
    """
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 2:
        return np.array([1.0, 0.0], dtype=np.float32), 1.0, 1.0, 0.0

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)  # [N,2], x-y
    mean = pts.mean(axis=0, keepdims=True)
    centered = pts - mean

    # 2x2 covariance
    cov = centered.T @ centered / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0].astype(np.float32)
    minor = eigvecs[:, 1].astype(np.float32)

    # project coordinates to major / minor axes
    proj_major = centered @ major
    proj_minor = centered @ minor

    length = float(proj_major.max() - proj_major.min() + 1.0)
    width = float(proj_minor.max() - proj_minor.min() + 1.0)
    angle_deg = _angle_deg_from_vec(major)

    return major, length, width, angle_deg


def _component_stats(
    label: int,
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    center_prob: np.ndarray,
    orient_map_hw2: np.ndarray,
    distance_map: Optional[np.ndarray],
) -> ComponentStats:
    x, y, w, h, area = stats[label].tolist()
    cx, cy = centroids[label].tolist()

    comp_mask = (labels == label)
    major_xy, length, width, angle_deg = _principal_axis_from_mask(comp_mask)
    aspect_ratio = float(length / max(width, 1e-6))

    center_vals = center_prob[comp_mask]
    center_mean = float(center_vals.mean()) if center_vals.size > 0 else 0.0

    orient_vals = orient_map_hw2[comp_mask]  # [N,2], ox,oy
    if orient_vals.size > 0:
        orient_vals = orient_vals / (np.linalg.norm(orient_vals, axis=1, keepdims=True) + 1e-6)
        dots = np.abs(orient_vals @ major_xy.reshape(2, 1)).reshape(-1)
        orient_consistency = float(dots.mean())
    else:
        orient_consistency = 0.0

    if distance_map is not None:
        dist_vals = distance_map[comp_mask]
        dist_mean = float(dist_vals.mean()) if dist_vals.size > 0 else np.nan
        dist_q80 = float(np.quantile(dist_vals, 0.8)) if dist_vals.size > 0 else np.nan
    else:
        dist_mean = np.nan
        dist_q80 = np.nan

    return ComponentStats(
        label=label,
        area=int(area),
        bbox=(int(x), int(y), int(w), int(h)),
        centroid=(float(cx), float(cy)),
        length=float(length),
        width=float(width),
        aspect_ratio=float(aspect_ratio),
        angle_deg=float(angle_deg),
        center_mean=float(center_mean),
        orient_consistency=float(orient_consistency),
        dist_mean=float(dist_mean),
        dist_q80=float(dist_q80),
    )


class GeometricVerifier:
    """
    Geometric verifier for suppressing short / isolated / off-direction false lines.

    Inputs:
        center_prob:   [H, W], float in [0,1]
        orient_map:    [2,H,W] or [H,W,2], unit or raw vectors
        distance_map:  optional [H, W], preferably normalized to [0,1]

    Main idea:
        1) threshold + connected components on center_prob
        2) local filtering:
           - area / length / aspect ratio
           - center confidence
           - orientation consistency
           - optional distance quality
        3) global dominant-direction filtering
        4) isolation filtering for short far-away components

    This is intentionally conservative and designed for your current
    "center + orient (+ optional distance)" pipeline.
    """

    def __init__(
        self,
        center_threshold: float = 0.30,
        morph_open_kernel: int = 3,
        morph_close_kernel: int = 3,
        min_area: int = 24,
        min_length: float = 30.0,
        min_aspect_ratio: float = 2.5,
        min_center_mean: float = 0.40,
        min_orient_consistency: float = 0.78,
        use_distance: bool = True,
        max_distance_mean: float = 0.30,
        max_distance_q80: float = 0.45,
        dominant_bin_deg: float = 15.0,
        dominant_keep_topk: int = 2,
        dominant_angle_tol_deg: float = 18.0,
        strong_length_px: float = 120.0,
        short_length_px: float = 100.0,
        max_isolated_dist_px: float = 220.0,
        debug: bool = True,
    ) -> None:
        self.center_threshold = float(center_threshold)
        self.morph_open_kernel = int(morph_open_kernel)
        self.morph_close_kernel = int(morph_close_kernel)
        self.min_area = int(min_area)
        self.min_length = float(min_length)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.min_center_mean = float(min_center_mean)
        self.min_orient_consistency = float(min_orient_consistency)
        self.use_distance = bool(use_distance)
        self.max_distance_mean = float(max_distance_mean)
        self.max_distance_q80 = float(max_distance_q80)
        self.dominant_bin_deg = float(dominant_bin_deg)
        self.dominant_keep_topk = int(dominant_keep_topk)
        self.dominant_angle_tol_deg = float(dominant_angle_tol_deg)
        self.strong_length_px = float(strong_length_px)
        self.short_length_px = float(short_length_px)
        self.max_isolated_dist_px = float(max_isolated_dist_px)
        self.debug = bool(debug)

    def _binarize_center(self, center_prob: np.ndarray) -> np.ndarray:
        mask = (center_prob >= self.center_threshold).astype(np.uint8)

        if self.morph_open_kernel > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_open_kernel, self.morph_open_kernel),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        if self.morph_close_kernel > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_close_kernel, self.morph_close_kernel),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        return mask

    def _local_keep(self, s: ComponentStats) -> Tuple[bool, str]:
        if s.area < self.min_area:
            return False, f'area<{self.min_area}'
        if s.length < self.min_length:
            return False, f'length<{self.min_length:.1f}'
        if s.aspect_ratio < self.min_aspect_ratio:
            return False, f'aspect<{self.min_aspect_ratio:.2f}'
        if s.center_mean < self.min_center_mean:
            return False, f'center_mean<{self.min_center_mean:.2f}'
        if s.orient_consistency < self.min_orient_consistency:
            return False, f'orient_consistency<{self.min_orient_consistency:.2f}'

        if self.use_distance and not np.isnan(s.dist_mean):
            if s.dist_mean > self.max_distance_mean:
                return False, f'dist_mean>{self.max_distance_mean:.2f}'
            if s.dist_q80 > self.max_distance_q80:
                return False, f'dist_q80>{self.max_distance_q80:.2f}'

        return True, 'keep'

    def _find_dominant_angles(self, stats_kept: List[ComponentStats]) -> List[float]:
        if len(stats_kept) == 0:
            return []

        bin_size = self.dominant_bin_deg
        n_bins = max(int(round(180.0 / bin_size)), 1)
        hist = np.zeros(n_bins, dtype=np.float32)

        # weighted by stronger, more reliable components
        for s in stats_kept:
            weight = float(
                max(s.length, 1.0) *
                max(s.center_mean, 1e-3) *
                max(s.orient_consistency, 1e-3)
            )
            bin_id = int(np.floor(s.angle_deg / bin_size)) % n_bins
            hist[bin_id] += weight

        top_ids = np.argsort(hist)[::-1][:self.dominant_keep_topk]
        dominant_angles = [(bid + 0.5) * bin_size for bid in top_ids if hist[bid] > 0]
        return [float(a % 180.0) for a in dominant_angles]

    def _dominant_keep(self, s: ComponentStats, dominant_angles: List[float]) -> Tuple[bool, str]:
        if len(dominant_angles) == 0:
            return True, 'no_dominant_filter'

        # long components are allowed a bit more freedom
        if s.length >= self.strong_length_px:
            return True, 'strong_component'

        diffs = [_angle_diff_deg(s.angle_deg, a) for a in dominant_angles]
        min_diff = min(diffs) if len(diffs) > 0 else 999.0
        if min_diff <= self.dominant_angle_tol_deg:
            return True, 'dominant_dir_ok'

        return False, f'off_dominant_dir>{self.dominant_angle_tol_deg:.1f}deg'

    def _isolation_keep(self, s: ComponentStats, strong_stats: List[ComponentStats]) -> Tuple[bool, str]:
        if s.length >= self.short_length_px:
            return True, 'not_short'

        if len(strong_stats) == 0:
            return True, 'no_strong_ref'

        cx, cy = s.centroid
        dists = []
        for ref in strong_stats:
            rx, ry = ref.centroid
            dists.append(np.hypot(cx - rx, cy - ry))
        min_dist = float(min(dists)) if len(dists) > 0 else 0.0

        if min_dist <= self.max_isolated_dist_px:
            return True, 'near_strong_group'

        return False, f'isolated>{self.max_isolated_dist_px:.1f}px'

    def verify(
        self,
        center_prob: np.ndarray,
        orient_map: np.ndarray,
        distance_map: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Args:
            center_prob:  [H, W] float [0,1]
            orient_map:   [2,H,W] or [H,W,2]
            distance_map: optional [H, W], normalized preferred

        Returns:
            dict with:
                mask: [H, W] uint8 {0,1}, refined mask
                raw_mask: [H, W] uint8 {0,1}, pre-verifier threshold mask
                components: list[dict]
                dominant_angles: list[float]
        """
        center_prob = _ensure_float01(center_prob)
        orient_map_hw2 = _normalize_orient(_ensure_orient_hw2(orient_map))

        if distance_map is not None:
            distance_map = np.asarray(distance_map, dtype=np.float32)
            if distance_map.shape != center_prob.shape:
                raise ValueError(
                    f'distance_map shape {distance_map.shape} must match center_prob shape {center_prob.shape}'
                )

        raw_mask = self._binarize_center(center_prob)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            raw_mask.astype(np.uint8), connectivity=8
        )

        all_stats: List[ComponentStats] = []
        local_kept: List[ComponentStats] = []
        local_keep_flags: Dict[int, Tuple[bool, str]] = {}

        for label in range(1, num_labels):  # 0 is background
            s = _component_stats(
                label=label,
                labels=labels,
                stats=stats,
                centroids=centroids,
                center_prob=center_prob,
                orient_map_hw2=orient_map_hw2,
                distance_map=distance_map,
            )
            all_stats.append(s)

            keep, reason = self._local_keep(s)
            local_keep_flags[label] = (keep, reason)
            if keep:
                local_kept.append(s)

        dominant_angles = self._find_dominant_angles(local_kept)

        # second pass: dominant direction
        dir_kept: List[ComponentStats] = []
        dir_keep_flags: Dict[int, Tuple[bool, str]] = {}
        for s in local_kept:
            keep, reason = self._dominant_keep(s, dominant_angles)
            dir_keep_flags[s.label] = (keep, reason)
            if keep:
                dir_kept.append(s)

        # strong references for isolation filtering
        strong_stats = [s for s in dir_kept if s.length >= self.strong_length_px]

        final_mask = np.zeros_like(raw_mask, dtype=np.uint8)
        final_flags: Dict[int, Tuple[bool, str]] = {}

        for s in dir_kept:
            keep, reason = self._isolation_keep(s, strong_stats)
            final_flags[s.label] = (keep, reason)
            if keep:
                final_mask[labels == s.label] = 1

        debug_components: List[Dict[str, object]] = []
        if self.debug:
            for s in all_stats:
                local_keep, local_reason = local_keep_flags.get(s.label, (False, 'not_checked'))
                dir_keep, dir_reason = dir_keep_flags.get(s.label, (False, 'not_checked'))
                fin_keep, fin_reason = final_flags.get(s.label, (False, 'filtered_before_final'))
                item = asdict(s)
                item.update(
                    local_keep=local_keep,
                    local_reason=local_reason,
                    dominant_keep=dir_keep,
                    dominant_reason=dir_reason,
                    final_keep=fin_keep,
                    final_reason=fin_reason,
                )
                debug_components.append(item)

        return dict(
            mask=final_mask.astype(np.uint8),
            raw_mask=raw_mask.astype(np.uint8),
            components=debug_components,
            dominant_angles=dominant_angles,
        )

    __call__ = verify