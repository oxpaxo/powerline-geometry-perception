from typing import Dict, List, Optional, Tuple

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

        if self.with_distance_head:
            has_dist_gt = hasattr(data_samples[0], 'gt_distance_map')
            if has_dist_gt:
                gt_distance = torch.stack(
                    [sample.gt_distance_map.data for sample in data_samples],
                    dim=0,
                ).float()
                dist_loss_dict = self.distance_head.loss(
                    distance_pred=pred_dict['distance_pred'],
                    gt_distance=gt_distance,
                )
                losses.update(dist_loss_dict)

        if self.with_tower_head:
            has_tower_gt = hasattr(data_samples[0], 'gt_tower_seg')
            if has_tower_gt:
                gt_tower = torch.stack(
                    [sample.gt_tower_seg.data for sample in data_samples],
                    dim=0,
                ).float()
                tower_h, tower_w = pred_dict['tower_pred'].shape[-2:]
                gt_tower_resized = F.interpolate(
                    gt_tower,
                    size=(tower_h, tower_w),
                    mode='nearest',
                )
                tower_loss_dict = self.tower_head.loss(
                    tower_logits=pred_dict['tower_pred'],
                    tower_gt=gt_tower_resized,
                )
                losses.update(tower_loss_dict)

        return losses

    def _upsample_pred_dict_to_input_size(
        self,
        pred_dict: Dict[str, torch.Tensor],
        input_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Upsample raw head outputs to input image size."""
        h, w = input_size
        out = {}

        out['center_logits'] = F.interpolate(
            pred_dict['center_logits'],
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        )
        out['orient_pred'] = F.interpolate(
            pred_dict['orient_pred'],
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        )

        if 'distance_pred' in pred_dict:
            out['distance_pred'] = F.interpolate(
                pred_dict['distance_pred'],
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )

        if 'tower_pred' in pred_dict:
            out['tower_pred'] = F.interpolate(
                pred_dict['tower_pred'],
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )

        return out

    def _whole_inference(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[SegDataSample]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Whole-image inference, then upsample outputs back to input size."""
        pred_dict = self._forward(inputs, data_samples)
        input_h, input_w = inputs.shape[-2:]
        return self._upsample_pred_dict_to_input_size(
            pred_dict,
            input_size=(input_h, input_w),
        )

    def _slide_inference(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[SegDataSample]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sliding-window inference on large images.

        test_cfg example:
            dict(
                mode='slide',
                crop_size=(512, 1024),
                stride=(384, 768),
                center_threshold=0.3,
            )
        """
        if inputs.size(0) != 1:
            raise NotImplementedError(
                'Sliding-window inference currently only supports batch_size=1.'
            )

        crop_size = self.test_cfg.get('crop_size', None)
        stride = self.test_cfg.get('stride', None)
        if crop_size is None or stride is None:
            raise ValueError(
                "Sliding-window inference requires test_cfg['crop_size'] and "
                "test_cfg['stride']."
            )

        crop_h, crop_w = crop_size
        stride_h, stride_w = stride

        n, _, img_h, img_w = inputs.shape
        device = inputs.device

        center_acc = torch.zeros((n, 1, img_h, img_w), device=device)
        orient_acc = torch.zeros((n, 2, img_h, img_w), device=device)

        distance_acc = None
        tower_acc = None
        if self.with_distance_head:
            distance_acc = torch.zeros((n, 1, img_h, img_w), device=device)
        if self.with_tower_head:
            tower_acc = torch.zeros((n, 1, img_h, img_w), device=device)

        count_mat = torch.zeros((n, 1, img_h, img_w), device=device)

        h_grids = max((img_h - crop_h + stride_h - 1) // stride_h + 1, 1)
        w_grids = max((img_w - crop_w + stride_w - 1) // stride_w + 1, 1)

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * stride_h
                x1 = w_idx * stride_w
                y2 = min(y1 + crop_h, img_h)
                x2 = min(x1 + crop_w, img_w)
                y1 = max(y2 - crop_h, 0)
                x1 = max(x2 - crop_w, 0)

                crop_img = inputs[:, :, y1:y2, x1:x2]

                crop_pred = self._forward(crop_img, data_samples=None)
                crop_pred = self._upsample_pred_dict_to_input_size(
                    crop_pred,
                    input_size=(y2 - y1, x2 - x1),
                )

                center_acc[:, :, y1:y2, x1:x2] += crop_pred['center_logits']
                orient_acc[:, :, y1:y2, x1:x2] += crop_pred['orient_pred']

                if self.with_distance_head and 'distance_pred' in crop_pred:
                    distance_acc[:, :, y1:y2, x1:x2] += crop_pred['distance_pred']

                if self.with_tower_head and 'tower_pred' in crop_pred:
                    tower_acc[:, :, y1:y2, x1:x2] += crop_pred['tower_pred']

                count_mat[:, :, y1:y2, x1:x2] += 1

        count_mat = count_mat.clamp_min(1.0)

        out = dict(
            center_logits=center_acc / count_mat,
            orient_pred=orient_acc / count_mat,
        )

        if distance_acc is not None:
            out['distance_pred'] = distance_acc / count_mat
        if tower_acc is not None:
            out['tower_pred'] = tower_acc / count_mat

        return out

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
        infer_mode = self.test_cfg.get('mode', 'whole')

        if infer_mode == 'slide':
            pred_dict = self._slide_inference(inputs, data_samples)
        else:
            pred_dict = self._whole_inference(inputs, data_samples)

        center_logits_up = pred_dict['center_logits']
        raw_orient_pred_up = pred_dict['orient_pred']

        center_prob = torch.sigmoid(center_logits_up)
        pred_orient_unit = self.orient_head.normalize_to_unit_vector(raw_orient_pred_up)

        center_threshold = float(self.test_cfg.get('center_threshold', 0.3))
        use_verifier = bool(self.test_cfg.get('use_geometric_verifier', False))

        if use_verifier:
            from projects.powerline_v1.utils.geometric_verifier import GeometricVerifier

            verifier_cfg = self.test_cfg.get('verifier_cfg', {})
            verifier = GeometricVerifier(
                center_threshold=center_threshold,
                **verifier_cfg
            )

            refined_masks = []
            batch_size = center_prob.shape[0]

            for i in range(batch_size):
                center_prob_np = center_prob[i, 0].detach().cpu().numpy()
                orient_np = pred_orient_unit[i].detach().cpu().numpy()  # [2,H,W]

                distance_np = None
                if 'distance_pred' in pred_dict:
                    distance_np = pred_dict['distance_pred'][i, 0].detach().cpu().numpy()

                verify_out = verifier(
                    center_prob=center_prob_np,
                    orient_map=orient_np,
                    distance_map=distance_np,
                )

                refined_mask = torch.from_numpy(verify_out['mask']).to(center_prob.device).long()
                refined_masks.append(refined_mask)

            pred_center_binary = torch.stack(refined_masks, dim=0).unsqueeze(1)
        else:
            pred_center_binary = (center_prob > center_threshold).long()

        results: List[SegDataSample] = []
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            sample = data_samples[i] if data_samples is not None else SegDataSample()
            sample.pred_sem_seg = PixelData(data=pred_center_binary[i])
            sample.seg_logits = PixelData(data=center_prob[i])
            sample.set_field(PixelData(data=pred_orient_unit[i]), 'pred_orient_map')

            if 'distance_pred' in pred_dict:
                sample.set_field(
                    PixelData(data=pred_dict['distance_pred'][i]),
                    'pred_distance_map'
                )

            if 'tower_pred' in pred_dict:
                sample.set_field(
                    PixelData(data=torch.sigmoid(pred_dict['tower_pred'][i])),
                    'pred_tower_map'
                )

            results.append(sample)

        return results