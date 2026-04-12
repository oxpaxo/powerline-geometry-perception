from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackPowerLineInputs(BaseTransform):
    """
    Pack image, centerline mask and orientation map into model inputs.
    """

    META_KEYS = (
        'img_path',
        'seg_map_path',
        'ori_shape',
        'img_shape',
        'pad_shape',
        'scale_factor',
        'flip',
        'flip_direction',
        'reduce_zero_label',
    )

    def __init__(self, meta_keys: Sequence[str] = META_KEYS) -> None:
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = {}

        img = results['img']
        if img.ndim < 3:
            img = np.expand_dims(img, -1)

        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)

        img = img.transpose(2, 0, 1)  # HWC -> CHW
        packed_results['inputs'] = to_tensor(img).float()

        data_sample = SegDataSample()

        if 'gt_seg_map' in results:
            gt_seg = results['gt_seg_map']
            if gt_seg.ndim == 2:
                gt_seg = gt_seg[None, ...]  # [1, H, W]
            gt_seg = to_tensor(np.ascontiguousarray(gt_seg)).long()
            data_sample.gt_sem_seg = PixelData(data=gt_seg)

        if 'gt_orient_map' in results:
            gt_orient = results['gt_orient_map']  # [H, W, 2]
            gt_orient = np.transpose(gt_orient, (2, 0, 1))  # [2, H, W]
            gt_orient = to_tensor(np.ascontiguousarray(gt_orient)).float()
            data_sample.set_field(
                PixelData(data=gt_orient),
                'gt_orient_map'
            )

        if 'gt_distance_map' in results:
            gt_dist = results['gt_distance_map']  # [H, W], float32
            if gt_dist.ndim == 2:
                gt_dist = gt_dist[None, ...]  # [1, H, W]
            gt_dist = to_tensor(np.ascontiguousarray(gt_dist)).float()
            data_sample.set_field(
                PixelData(data=gt_dist),
                'gt_distance_map'
            )

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(meta_keys={self.meta_keys})'