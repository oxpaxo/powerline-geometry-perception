from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class DistanceFieldHead(nn.Module):
    """
    Lightweight auxiliary head for distance field supervision.

    Input:
        shared_feat: [N, C, H, W]

    Output:
        distance_pred: [N, 1, H, W]  (raw, non-negative after ReLU)

    The distance target is the per-pixel distance to the nearest wire centerline
    pixel, clipped and optionally normalized. This auxiliary supervision helps
    the shared feature learn continuous geometric proximity to wires, reducing
    false positives on wire-like backgrounds.

    Loss:
        SmoothL1 (default) or L1.

        When use_mask=True:
            - only samples that contain at least one non-zero distance pixel
              participate in the loss
            - within each valid sample, the loss is computed on the full map

        This avoids the previous inconsistency where the numerator was computed
        on almost all pixels but the denominator only counted pixels with
        gt_distance > 0.
    """

    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 64,
        loss_distance_weight: float = 0.5,
        target_normalize_mode: Literal['none', 'max', 'clip_max'] = 'clip_max',
        max_distance_clip: float = 50.0,
        use_mask: bool = True,
        loss_type: Literal['smooth_l1', 'l1'] = 'smooth_l1',
        # Reserved for future attraction field extension (task_type='attraction')
        task_type: Literal['distance', 'attraction'] = 'distance',
    ) -> None:
        super().__init__()
        assert task_type == 'distance', (
            "task_type='attraction' is reserved for future extension. "
            "Only 'distance' is implemented in this version."
        )

        self.loss_distance_weight = loss_distance_weight
        self.target_normalize_mode = target_normalize_mode
        self.max_distance_clip = max_distance_clip
        self.use_mask = use_mask
        self.loss_type = loss_type
        self.task_type = task_type

        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),  # distance is non-negative
        )

    def forward(self, shared_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_feat: [N, C, H, W]

        Returns:
            distance_pred: [N, 1, H, W], non-negative
        """
        return self.conv_head(shared_feat)

    def _normalize_target(self, gt_distance: torch.Tensor) -> torch.Tensor:
        """
        Clip and/or normalize the distance target tensor.

        Args:
            gt_distance: [N, 1, H, W], raw pixel distance values

        Returns:
            normalized distance: [N, 1, H, W]
        """
        if self.target_normalize_mode == 'none':
            return gt_distance

        clipped = gt_distance.clamp(max=self.max_distance_clip)

        if self.target_normalize_mode == 'clip_max':
            return clipped / self.max_distance_clip

        if self.target_normalize_mode == 'max':
            batch_max = clipped.flatten(1).max(dim=1)[0].clamp_min(1.0)
            return clipped / batch_max.view(-1, 1, 1, 1)

        raise ValueError(
            f'Unsupported target_normalize_mode: {self.target_normalize_mode}'
        )

    def _build_sample_mask(self, gt_distance: torch.Tensor) -> torch.Tensor:
        """
        Build a per-sample valid mask.

        A sample is considered valid if it contains at least one non-zero
        distance pixel, i.e. it is not an all-background/no-wire patch.

        Args:
            gt_distance: [N, 1, H, W]

        Returns:
            pixel_mask: [N, 1, H, W], float tensor in {0,1}
        """
        sample_has_wire = (gt_distance.flatten(1).max(dim=1)[0] > 0).float()
        pixel_mask = sample_has_wire.view(-1, 1, 1, 1).expand_as(gt_distance)
        return pixel_mask

    def loss(
        self,
        distance_pred: torch.Tensor,
        gt_distance: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            distance_pred: [N, 1, H, W]
            gt_distance:   [N, 1, H, W], raw pixel-distance values

        Returns:
            dict with:
                - loss_distance
                - loss_distance_raw
        """
        out_h, out_w = distance_pred.shape[-2:]
        if gt_distance.shape[-2:] != (out_h, out_w):
            gt_distance = F.interpolate(
                gt_distance,
                size=(out_h, out_w),
                mode='bilinear',
                align_corners=False,
            )

        gt_norm = self._normalize_target(gt_distance)

        if self.loss_type == 'l1':
            raw_loss_map = F.l1_loss(distance_pred, gt_norm, reduction='none')
        else:
            raw_loss_map = F.smooth_l1_loss(
                distance_pred,
                gt_norm,
                reduction='none',
            )

        if self.use_mask:
            pixel_mask = self._build_sample_mask(gt_distance)
        else:
            pixel_mask = torch.ones_like(gt_norm)

        valid_pixels = pixel_mask.sum().clamp_min(1.0)

        loss_raw = (raw_loss_map * pixel_mask).sum() / valid_pixels
        loss_distance = self.loss_distance_weight * loss_raw

        return dict(
            loss_distance=loss_distance,
            loss_distance_raw=loss_raw.detach(),
        )