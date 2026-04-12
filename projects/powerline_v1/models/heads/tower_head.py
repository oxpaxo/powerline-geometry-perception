"""
TowerHead — auxiliary head for tower/pylon segmentation.

STATUS: SKELETON ONLY — not yet trainable.

WHY NOT FULLY IMPLEMENTED:
    The current local workspace does not contain tower/pylon annotation masks.
    The TTPLA converter (convert_ttpla_to_v1.py) only processes `cable` label
    shapes; no tower polygon extraction is implemented. To enable this head you
    need either:
        1. LabelMe JSON annotations that include a 'tower' or 'pylon' label,
           AND an updated converter that writes tower/*.png masks.
        2. Or: an alternative tower mask dataset in the same spatial layout.

HOW TO ENABLE (future work):
    1. Update convert_ttpla_to_v1.py to also extract tower polygons and write
       tower/{train,val}/{stem}.png binary masks.
    2. Update LoadPowerLineAnnotations to also load gt_tower_seg.
    3. Update PowerLineRandomCrop / PowerLineRandomFlip to sync gt_tower_seg.
    4. Update PackPowerLineInputs to pack gt_tower_seg into SegDataSample.
    5. Set tower_head=dict(type='TowerHead', ...) in config.
    6. The PowerLineSegmentor already has with_tower_head / tower_head support.

INTERFACE (when data is available):
    Input:  shared_feat [N, C, H, W]
    Output: tower_logits [N, 1, H, W]
    Loss:   BCE + Dice (same pattern as CenterlineHead)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class TowerHead(nn.Module):
    """
    Binary segmentation head for tower/pylon auxiliary supervision.

    Mirrors the architecture of CenterlineHead (BCE + Dice).
    Will raise NotImplementedError if you attempt to call loss() without
    providing real tower GT — a safe guard against accidental use.
    """

    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 128,
        loss_bce_weight: float = 1.0,
        loss_dice_weight: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.loss_bce_weight = loss_bce_weight
        self.loss_dice_weight = loss_dice_weight
        self.eps = eps

        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, shared_feat: torch.Tensor) -> torch.Tensor:
        return self.conv_head(shared_feat)

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits).reshape(logits.size(0), -1)
        target = target.reshape(target.size(0), -1).float()
        inter = (prob * target).sum(dim=1)
        denom = prob.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()

    def loss(
        self,
        tower_logits: torch.Tensor,
        tower_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tower_logits: [N, 1, H, W]
            tower_gt:     [N, 1, H, W], float32 binary

        Returns:
            dict with loss_tower, loss_tower_bce, loss_tower_dice
        """
        if tower_gt.dim() == 3:
            tower_gt = tower_gt.unsqueeze(1)
        tower_gt = tower_gt.float()

        loss_tower_bce = F.binary_cross_entropy_with_logits(tower_logits, tower_gt)
        loss_tower_dice = self._dice_loss(tower_logits, tower_gt)
        loss_tower = (
            self.loss_bce_weight * loss_tower_bce
            + self.loss_dice_weight * loss_tower_dice
        )
        return dict(
            loss_tower=loss_tower,
            loss_tower_bce=loss_tower_bce.detach(),
            loss_tower_dice=loss_tower_dice.detach(),
        )
