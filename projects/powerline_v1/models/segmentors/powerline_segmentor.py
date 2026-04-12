from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


@MODELS.register_module()
class PowerLineSegmentor(BaseSegmentor):
    """
    Two-head powerline segmentor.

    Current supported feature routes:
        1) Image -> Backbone -> Neck -> SharedFusion -> {CenterlineHead, OrientationHead}
        2) Image -> Backbone -> SharedFusion -> {CenterlineHead, OrientationHead}

    This keeps the old ResNet18 + FPN path intact, while allowing a new
    SegFormer/MixVisionTransformer path without an explicit neck.
    """

    def __init__(
        self,
        backbone: dict,
        neck: Optional[dict] = None,
        fusion: Optional[dict] = None,
        center_head: Optional[dict] = None,
        orient_head: Optional[dict] = None,
        distance_head: Optional[dict] = None,
        tower_head: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if fusion is None:
            raise ValueError('fusion config must not be None')
        if center_head is None:
            raise ValueError('center_head config must not be None')
        if orient_head is None:
            raise ValueError('orient_head config must not be None')

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.fusion = MODELS.build(fusion)
        self.center_head = MODELS.build(center_head)
        self.orient_head = MODELS.build(orient_head)
        self.distance_head = MODELS.build(distance_head) if distance_head is not None else None
        self.tower_head = MODELS.build(tower_head) if tower_head is not None else None

        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or dict(center_threshold=0.3)

    @property
    def with_neck(self) -> bool:
        return self.neck is not None

    @property
    def with_distance_head(self) -> bool:
        return self.distance_head is not None

    @property
    def with_tower_head(self) -> bool:
        return self.tower_head is not None

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, 3, H, W]

        Returns:
            shared_feat: [N, C, H_out, W_out]
        """
        backbone_feats = self.backbone(inputs)

        if not isinstance(backbone_feats, (list, tuple)):
            raise TypeError(
                f'backbone must return list/tuple of multi-scale features, '
                f'but got {type(backbone_feats)}'
            )

        feats = self.neck(backbone_feats) if self.with_neck else backbone_feats
        shared_feat = self.fusion(feats)
        return shared_feat

    def encode_decode(
        self,
        inputs: torch.Tensor,
        batch_img_metas: Optional[List[dict]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Required by BaseSegmentor.

        Args:
            inputs: [N, 3, H, W]
            batch_img_metas: unused here, kept for interface compatibility

        Returns:
            dict with:
                center_logits: [N, 1, H_out, W_out]
                orient_pred:   [N, 2, H_out, W_out]
                distance_pred: [N, 1, H_out, W_out]  (only when with_distance_head)
                tower_pred:    [N, 1, H_out, W_out]  (only when with_tower_head)
        """
        shared_feat = self.extract_feat(inputs)

        out = dict(
            center_logits=self.center_head(shared_feat),
            orient_pred=self.orient_head(shared_feat),
        )
        if self.with_distance_head:
            out['distance_pred'] = self.distance_head(shared_feat)
        if self.with_tower_head:
            out['tower_pred'] = self.tower_head(shared_feat)
        return out

    def _forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[SegDataSample]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tensor mode forward, no post-processing."""
        batch_img_metas = None
        if data_samples is not None:
            batch_img_metas = [sample.metainfo for sample in data_samples]
        return self.encode_decode(inputs, batch_img_metas=batch_img_metas)

    def loss(
        self,
        inputs: torch.Tensor,
        data_samples: List[SegDataSample],
    ) -> Dict[str, torch.Tensor]:
        """Loss mode forward."""
        pred_dict = self._forward(inputs, data_samples)
        center_logits = pred_dict['center_logits']
        raw_orient_pred = pred_dict['orient_pred']

        gt_center = torch.stack(
            [sample.gt_sem_seg.data for sample in data_samples],
            dim=0,
        ).float()

        gt_orient = torch.stack(
            [sample.gt_orient_map.data for sample in data_samples],
            dim=0,
        ).float()

        out_h, out_w = center_logits.shape[-2:]
        gt_center_resized = F.interpolate(
            gt_center,
            size=(out_h, out_w),
            mode='nearest',
        )

        gt_orient_resized = F.interpolate(
            gt_orient,
            size=(out_h, out_w),
            mode='bilinear',
            align_corners=False,
        )

        gt_orient_resized = self.orient_head.normalize_to_unit_vector(gt_orient_resized)
        gt_orient_resized = gt_orient_resized * gt_center_resized.float()

        losses = {}

        center_loss_dict = self.center_head.loss(center_logits, gt_center_resized)
        losses.update(center_loss_dict)

        orient_loss_dict = self.orient_head.loss(
            raw_orient_pred=raw_orient_pred,
            orient_gt=gt_orient_resized,
            valid_mask=gt_center_resized,
        )
        losses.update(orient_loss_dict)

        # Distance auxiliary head
        if self.with_distance_head and hasattr(data_samples[0], 'gt_distance_map'):
            distance_pred = pred_dict['distance_pred']
            gt_distance = torch.stack(
                [sample.gt_distance_map.data for sample in data_samples],
                dim=0,
            ).float()
            distance_loss_dict = self.distance_head.loss(distance_pred, gt_distance)
            losses.update(distance_loss_dict)

        # Tower auxiliary head (placeholder — requires gt_tower_seg in data_samples)
        if self.with_tower_head and hasattr(data_samples[0], 'gt_tower_seg'):
            tower_pred = pred_dict['tower_pred']
            gt_tower = torch.stack(
                [sample.gt_tower_seg.data for sample in data_samples],
                dim=0,
            ).float()
            tower_loss_dict = self.tower_head.loss(tower_pred, gt_tower)
            losses.update(tower_loss_dict)

        return losses

    def predict(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[SegDataSample]] = None,
    ) -> List[SegDataSample]:
        """
        Predict mode forward.
        Returns a list of SegDataSample with:
            - pred_sem_seg
            - seg_logits
            - pred_orient_map
        """
        pred_dict = self._forward(inputs, data_samples)
        center_logits = pred_dict['center_logits']
        raw_orient_pred = pred_dict['orient_pred']

        input_h, input_w = inputs.shape[-2:]

        center_logits_up = F.interpolate(
            center_logits,
            size=(input_h, input_w),
            mode='bilinear',
            align_corners=False,
        )
        raw_orient_pred_up = F.interpolate(
            raw_orient_pred,
            size=(input_h, input_w),
            mode='bilinear',
            align_corners=False,
        )

        center_prob = torch.sigmoid(center_logits_up)
        pred_orient_unit = self.orient_head.normalize_to_unit_vector(raw_orient_pred_up)

        center_threshold = float(self.test_cfg.get('center_threshold', 0.3))
        pred_center_binary = (center_prob > center_threshold).long()

        results: List[SegDataSample] = []
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            sample = data_samples[i] if data_samples is not None else SegDataSample()
            sample.pred_sem_seg = PixelData(data=pred_center_binary[i])
            sample.seg_logits = PixelData(data=center_prob[i])
            sample.set_field(PixelData(data=pred_orient_unit[i]), 'pred_orient_map')
            results.append(sample)

        return results