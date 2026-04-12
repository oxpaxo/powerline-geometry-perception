# projects/powerline_v1/models/heads/orientation_head.py

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class OrientationHead(nn.Module):
    """
    Orientation head.

    Input:
        shared feature: [N, C, H, W]

    Output:
        raw orientation prediction: [N, 2, H, W]

    Channel meaning:
        channel 0 -> ox
        channel 1 -> oy
    """

    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 128,
        loss_smoothl1_weight: float = 1.0,
        loss_cosine_weight: float = 0.2,
    ):
        super().__init__()

        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1, bias=True),
        )

        self.loss_smoothl1_weight = loss_smoothl1_weight
        self.loss_cosine_weight = loss_cosine_weight

    def forward(self, shared_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_feat: [N, C, H, W]

        Returns:
            raw_orient_pred: [N, 2, H, W]
        """
        raw_orient_pred = self.conv_head(shared_feat)
        return raw_orient_pred

    @staticmethod
    def normalize_to_unit_vector(orient_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            orient_map: [N, 2, H, W]

        Returns:
            unit_orient_map: [N, 2, H, W]
        """
        norm = torch.norm(orient_map, dim=1, keepdim=True) + 1e-6
        unit_orient_map = orient_map / norm
        return unit_orient_map

    def loss(
        self,
        raw_orient_pred: torch.Tensor,
        orient_gt: torch.Tensor, # 方向真值图
        valid_mask: torch.Tensor, # 中心线区域掩膜，仅在该区域计算损失
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            raw_orient_pred: [N, 2, H, W]
            orient_gt:       [N, 2, H, W]
            valid_mask:      [N, 1, H, W], only centerline area contributes to loss

        Returns:
            dict of orientation losses
        """
        pred_unit = self.normalize_to_unit_vector(raw_orient_pred)
        gt_unit = self.normalize_to_unit_vector(orient_gt)

        mask_2ch = valid_mask.float().repeat(1, 2, 1, 1) # 这里将单通道掩膜复制成 2 通道（对齐 ox/oy）

        # 1) masked SmoothL1
        smooth_l1_map = F.smooth_l1_loss(
            pred_unit, gt_unit, reduction="none"
        )  # [N, 2, H, W]
        smooth_l1_map = smooth_l1_map * mask_2ch
        valid_elements = mask_2ch.sum().clamp_min(1.0)
        loss_orient_smoothl1 = smooth_l1_map.sum() / valid_elements # 方向回归损失

        # 2) masked cosine consistency
        cosine_sim = (pred_unit * gt_unit).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        cosine_loss_map = (1.0 - cosine_sim) * valid_mask.float()
        valid_pixels = valid_mask.float().sum().clamp_min(1.0)
        loss_orient_cosine = cosine_loss_map.sum() / valid_pixels # 方向一致性损失

        # total
        loss_orient = (
            self.loss_smoothl1_weight * loss_orient_smoothl1
            + self.loss_cosine_weight * loss_orient_cosine
        )

        return dict(
            loss_orient=loss_orient,
            loss_orient_smoothl1=loss_orient_smoothl1.detach(),
            loss_orient_cosine=loss_orient_cosine.detach(),
        )
