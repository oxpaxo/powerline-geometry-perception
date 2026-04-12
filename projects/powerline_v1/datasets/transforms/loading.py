from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


def _compute_distance_map(center_mask: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel L2 distance to the nearest wire centerline pixel.

    Args:
        center_mask: [H, W], uint8, binary (>0 means centerline)

    Returns:
        distance_map: [H, W], float32, pixel distances (0 on centerline)
    """
    import cv2
    wire_binary = (center_mask > 0).astype(np.uint8)
    if wire_binary.sum() == 0:
        # No wire in this sample — all pixels are at max distance.
        # Return zeros; the loss mask will handle this gracefully.
        return np.zeros(center_mask.shape, dtype=np.float32)
    # distanceTransform computes distance from non-zero pixels.
    # We want distance FROM background pixels TO the wire.
    bg_mask = (1 - wire_binary).astype(np.uint8)
    dist = cv2.distanceTransform(bg_mask, cv2.DIST_L2, maskSize=5)
    # Wire pixels themselves get distance 0.
    dist[wire_binary > 0] = 0.0
    return dist.astype(np.float32)


@TRANSFORMS.register_module()
class LoadPowerLineAnnotations(BaseTransform):
    """
    Load centerline png and orientation npy.

    Outputs:
        results['gt_seg_map']      : [H, W], uint8, binary
        results['gt_orient_map']   : [H, W, 2], float32
        results['gt_distance_map'] : [H, W], float32  (only when generate_distance_map=True)
    """

    def __init__(
        self,
        to_float32: bool = True,
        binary_center: bool = True,
        generate_distance_map: bool = False,
    ) -> None:
        self.to_float32 = to_float32
        self.binary_center = binary_center
        self.generate_distance_map = generate_distance_map

    def transform(self, results: dict) -> Optional[dict]:
        seg_map_path = results['seg_map_path']
        orient_path = results['orient_path']

        center = mmcv.imread(seg_map_path, flag='grayscale')
        if center is None:
            raise FileNotFoundError(f'Failed to load centerline map: {seg_map_path}')

        if self.binary_center:
            center = (center > 0).astype(np.uint8)

        orient = np.load(orient_path)
        if orient.ndim != 3 or orient.shape[2] != 2:
            raise ValueError(
                f'Orientation map must be [H, W, 2], but got {orient.shape}'
            )

        if self.to_float32:
            orient = orient.astype(np.float32)

        if center.shape[:2] != orient.shape[:2]:
            raise ValueError(
                'Shape mismatch between centerline map and orient map: '
                f'{center.shape[:2]} vs {orient.shape[:2]}'
            )

        results['gt_seg_map'] = center
        results['gt_orient_map'] = orient
        results['seg_fields'].append('gt_seg_map')

        if self.generate_distance_map:
            results['gt_distance_map'] = _compute_distance_map(center)

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f'binary_center={self.binary_center}, '
                f'generate_distance_map={self.generate_distance_map})')