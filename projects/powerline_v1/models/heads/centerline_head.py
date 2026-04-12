from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class CenterlineHead(nn.Module):
    """
    Centerline prediction head.

    Input:
        shared feature: [N, C, H, W]

    Output:
        center logits: [N, 1, H, W]
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

    @staticmethod
    def predict_prob(center_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(center_logits)

    def _dice_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prob = torch.sigmoid(logits)

        prob = prob.reshape(prob.size(0), -1)
        target = target.reshape(target.size(0), -1).float()

        inter = (prob * target).sum(dim=1)
        denom = prob.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()

    def loss(
        self,
        center_logits: torch.Tensor,
        center_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if center_gt.dim() == 3:
            center_gt = center_gt.unsqueeze(1)

        center_gt = center_gt.float()

        loss_center_bce = F.binary_cross_entropy_with_logits(
            center_logits,
            center_gt,
        )
        loss_center_dice = self._dice_loss(center_logits, center_gt)

        loss_center = (
            self.loss_bce_weight * loss_center_bce
            + self.loss_dice_weight * loss_center_dice
        )

        return dict(
            loss_center=loss_center,
            loss_center_bce=loss_center_bce.detach(),
            loss_center_dice=loss_center_dice.detach(),
        )