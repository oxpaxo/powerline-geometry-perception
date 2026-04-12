from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class SharedFusion(nn.Module):
    """
    Fuse multi-scale features into one shared feature map.

    This version is backward compatible with the old FPN setup and also supports
    heterogeneous channel dimensions from transformer encoders such as
    SegFormer/MixVisionTransformer.

    Supported config styles:
        1) SharedFusion(in_channels=128, num_levels=4, out_channels=128)
        2) SharedFusion(in_channels_list=[64, 128, 320, 512], out_channels=128)

    Input:
        feats = [feat0, feat1, feat2, feat3]
        each feat_i: [N, C_i, H_i, W_i]

    Output:
        shared_feat: [N, out_channels, H0, W0]
        where H0, W0 are from feats[0]
    """

    def __init__(
        self,
        in_channels: Optional[int] = None,
        in_channels_list: Optional[Sequence[int]] = None,
        num_levels: Optional[int] = None,
        out_channels: int = 128,
        use_bn: bool = True,
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        if in_channels_list is None:
            if in_channels is None or num_levels is None:
                raise ValueError(
                    'Either provide (in_channels and num_levels), or provide '
                    'in_channels_list.'
                )
            in_channels_list = [in_channels] * num_levels
        else:
            in_channels_list = list(in_channels_list)
            if len(in_channels_list) == 0:
                raise ValueError('in_channels_list must not be empty')
            if num_levels is not None and len(in_channels_list) != num_levels:
                raise ValueError(
                    f'num_levels={num_levels} but len(in_channels_list)='
                    f'{len(in_channels_list)}'
                )
            num_levels = len(in_channels_list)

        self.in_channels_list = list(in_channels_list)
        self.num_levels = int(num_levels)
        self.out_channels = out_channels
        self.align_corners = align_corners

        self.level_adapters = nn.ModuleList()
        for in_ch in self.in_channels_list:
            layers = [
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.level_adapters.append(nn.Sequential(*layers))

        fuse_in_channels = out_channels * self.num_levels
        fuse_layers = [
            nn.Conv2d(
                fuse_in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        ]
        if use_bn:
            fuse_layers.append(nn.BatchNorm2d(out_channels))
        fuse_layers.append(nn.ReLU(inplace=True))
        self.fuse_conv = nn.Sequential(*fuse_layers)

        self.out_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(feats, (list, tuple)):
            raise TypeError(f'feats must be list/tuple, but got {type(feats)}')
        if len(feats) != self.num_levels:
            raise ValueError(
                f'Expected {self.num_levels} feature levels, but got {len(feats)}'
            )

        ref_feat = feats[0]
        target_size = ref_feat.shape[-2:]

        resized_feats = []
        for idx, feat in enumerate(feats):
            expected_in_channels = self.in_channels_list[idx]
            if feat.shape[1] != expected_in_channels:
                raise ValueError(
                    f'Feature level {idx} expects {expected_in_channels} channels, '
                    f'but got {feat.shape[1]}'
                )

            feat = self.level_adapters[idx](feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                )
            resized_feats.append(feat)

        x = torch.cat(resized_feats, dim=1)
        x = self.fuse_conv(x)
        x = self.out_conv(x)
        return x