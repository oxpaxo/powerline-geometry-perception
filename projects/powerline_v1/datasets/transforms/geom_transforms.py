# projects/powerline_v1/datasets/transforms/geom_transforms.py

import random
from typing import Optional, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


def renormalize_orientation_field(
    orient_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Args:
        orient_map: [H, W, 2], float32
        valid_mask: [H, W], non-zero means valid centerline pixels

    Returns:
        normalized_orient_map: [H, W, 2], float32
    """
    out = orient_map.astype(np.float32).copy()
    ox = out[..., 0]
    oy = out[..., 1]

    norm = np.sqrt(ox * ox + oy * oy) + 1e-6
    ox = ox / norm
    oy = oy / norm

    if valid_mask is not None:
        invalid = valid_mask <= 0
        ox[invalid] = 0.0
        oy[invalid] = 0.0

    out[..., 0] = ox
    out[..., 1] = oy
    return out


def enforce_upper_half_plane(
    orient_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Enforce the same half-plane convention as dataset converter:
    if oy < 0, flip the whole vector.

    Args:
        orient_map: [H, W, 2]
        valid_mask: [H, W]

    Returns:
        adjusted_orient_map: [H, W, 2]
    """
    out = orient_map.astype(np.float32).copy()
    ox = out[..., 0]
    oy = out[..., 1]

    if valid_mask is None:
        valid = np.ones_like(ox, dtype=bool)
    else:
        valid = valid_mask > 0

    need_flip = valid & (oy < 0)
    ox[need_flip] *= -1.0
    oy[need_flip] *= -1.0

    out[..., 0] = ox
    out[..., 1] = oy
    return out


def compute_distance_map_from_center(center_mask: np.ndarray) -> np.ndarray:
    """
    Recompute per-pixel L2 distance to the nearest wire centerline pixel.

    Args:
        center_mask: [H, W], uint8/binary, >0 means centerline

    Returns:
        distance_map: [H, W], float32
    """
    wire_binary = (center_mask > 0).astype(np.uint8)

    if wire_binary.sum() == 0:
        return np.zeros(center_mask.shape, dtype=np.float32)

    bg_mask = (1 - wire_binary).astype(np.uint8)
    dist = cv2.distanceTransform(bg_mask, cv2.DIST_L2, maskSize=5)
    dist[wire_binary > 0] = 0.0
    return dist.astype(np.float32)


@TRANSFORMS.register_module()
class PowerLineRandomCrop(BaseTransform):
    """
    Crop image, centerline GT, orientation GT, and optional distance GT together.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int],
        min_positive_pixels: int = 0,
        max_try: int = 10,
        pad_if_needed: bool = True,
    ):
        self.crop_size = crop_size
        self.min_positive_pixels = min_positive_pixels
        self.max_try = max_try
        self.pad_if_needed = pad_if_needed

    def _pad_to_crop_size(
        self,
        img: np.ndarray,
        seg_map: np.ndarray,
        orient_map: np.ndarray,
        distance_map: Optional[np.ndarray] = None,
    ):
        crop_h, crop_w = self.crop_size
        img_h, img_w = img.shape[:2]

        pad_h = max(0, crop_h - img_h)
        pad_w = max(0, crop_w - img_w)

        if pad_h == 0 and pad_w == 0:
            return img, seg_map, orient_map, distance_map

        img = np.pad(
            img,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        seg_map = np.pad(
            seg_map,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )
        orient_map = np.pad(
            orient_map,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        if distance_map is not None:
            distance_map = np.pad(
                distance_map,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        return img, seg_map, orient_map, distance_map

    def transform(self, results: dict) -> dict:
        img = results["img"]
        seg_map = results["gt_seg_map"]
        orient_map = results["gt_orient_map"]
        distance_map = results.get("gt_distance_map", None)

        if self.pad_if_needed:
            img, seg_map, orient_map, distance_map = self._pad_to_crop_size(
                img, seg_map, orient_map, distance_map
            )

        img_h, img_w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        assert img_h >= crop_h and img_w >= crop_w, (
            f"crop_size={self.crop_size} is larger than current image size={(img_h, img_w)}"
        )

        final_top = 0
        final_left = 0

        for _ in range(self.max_try):
            top = random.randint(0, img_h - crop_h)
            left = random.randint(0, img_w - crop_w)

            seg_patch = seg_map[top: top + crop_h, left: left + crop_w]
            positive_pixels = int((seg_patch > 0).sum())

            final_top = top
            final_left = left

            if positive_pixels >= self.min_positive_pixels:
                break

        top = final_top
        left = final_left

        img = img[top: top + crop_h, left: left + crop_w]
        seg_map = seg_map[top: top + crop_h, left: left + crop_w]
        orient_map = orient_map[top: top + crop_h, left: left + crop_w, :]

        results["img"] = img
        results["gt_seg_map"] = seg_map
        results["gt_orient_map"] = orient_map
        results["img_shape"] = img.shape[:2]
        results["crop_bbox"] = np.array(
            [top, left, top + crop_h, left + crop_w], dtype=np.int32
        )

        if distance_map is not None:
            results["gt_distance_map"] = distance_map[
                top: top + crop_h, left: left + crop_w
            ]

        return results


@TRANSFORMS.register_module()
class PowerLineRandomFlip(BaseTransform):
    """
    V1: only horizontal flip is recommended.

    horizontal:
        img         -> flip left-right
        seg_map     -> flip left-right
        orient_map  -> flip left-right, and ox *= -1

    vertical:
        supported here, but not recommended for V1
    """

    def __init__(self, prob: float = 0.5, direction: str = "horizontal"):
        assert direction in ["horizontal", "vertical"]
        self.prob = prob
        self.direction = direction

    def transform(self, results: dict) -> dict:
        if random.random() >= self.prob:
            results["flip"] = False
            results["flip_direction"] = None
            return results

        img = results["img"]
        seg_map = results["gt_seg_map"]
        orient_map = results["gt_orient_map"]
        distance_map = results.get("gt_distance_map", None)

        if self.direction == "horizontal":
            img = np.flip(img, axis=1).copy()
            seg_map = np.flip(seg_map, axis=1).copy()
            orient_map = np.flip(orient_map, axis=1).copy()
            orient_map[..., 0] *= -1.0

            if distance_map is not None:
                distance_map = np.flip(distance_map, axis=1).copy()

        else:
            img = np.flip(img, axis=0).copy()
            seg_map = np.flip(seg_map, axis=0).copy()
            orient_map = np.flip(orient_map, axis=0).copy()
            orient_map[..., 1] *= -1.0
            orient_map = enforce_upper_half_plane(orient_map, seg_map)

            if distance_map is not None:
                distance_map = np.flip(distance_map, axis=0).copy()

        orient_map = renormalize_orientation_field(orient_map, seg_map)

        results["img"] = img
        results["gt_seg_map"] = seg_map
        results["gt_orient_map"] = orient_map
        results["flip"] = True
        results["flip_direction"] = self.direction

        if distance_map is not None:
            results["gt_distance_map"] = distance_map

        return results


@TRANSFORMS.register_module()
class PowerLineResize(BaseTransform):
    """
    Resize image, centerline GT, orientation GT, and optional distance GT together.

    For orientation GT:
    - resize each channel
    - renormalize to unit vectors
    - re-enforce the half-plane convention

    For distance GT:
    - do NOT simply resize the previous distance map
    - instead recompute it from the resized centerline mask
      so the geometry stays self-consistent
    """

    def __init__(self, scale: Tuple[int, int], keep_ratio: bool = False):
        self.scale = scale  # (target_w, target_h)
        self.keep_ratio = keep_ratio

    def transform(self, results: dict) -> dict:
        img = results["img"]
        seg_map = results["gt_seg_map"]
        orient_map = results["gt_orient_map"]
        has_distance_map = "gt_distance_map" in results

        old_h, old_w = img.shape[:2]
        target_w, target_h = self.scale

        if self.keep_ratio:
            ratio = min(target_w / old_w, target_h / old_h)
            new_w = int(round(old_w * ratio))
            new_h = int(round(old_h * ratio))
        else:
            new_w, new_h = target_w, target_h

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        seg_map = cv2.resize(seg_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        ox = cv2.resize(
            orient_map[..., 0], (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        oy = cv2.resize(
            orient_map[..., 1], (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        orient_map = np.stack([ox, oy], axis=-1).astype(np.float32)

        orient_map = renormalize_orientation_field(orient_map, seg_map)
        orient_map = enforce_upper_half_plane(orient_map, seg_map)

        results["img"] = img
        results["gt_seg_map"] = seg_map
        results["gt_orient_map"] = orient_map
        results["img_shape"] = img.shape[:2]
        results["scale_factor"] = (new_w / old_w, new_h / old_h)

        if has_distance_map:
            results["gt_distance_map"] = compute_distance_map_from_center(seg_map)

        return results