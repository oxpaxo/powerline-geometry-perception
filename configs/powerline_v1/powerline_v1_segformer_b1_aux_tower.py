"""
Tower auxiliary head config (PLACEHOLDER — no tower GT available yet).

TowerHead is registered and built, but loss() will NOT be called during
training because data_samples will not contain gt_tower_seg.
This config exists to verify that the model builds correctly with tower_head
wired in, and to serve as the ready-to-use config once tower annotations
become available.

To enable actual tower supervision:
    1. Update convert_ttpla_to_v1.py to extract tower polygons → tower/*.png
    2. Update LoadPowerLineAnnotations to load gt_tower_seg
    3. Update geom_transforms to sync gt_tower_seg with crop/flip
    4. Update PackPowerLineInputs to pack gt_tower_seg
    Then tower_head loss will automatically activate (see powerline_segmentor.py).
"""
_base_ = ['./powerline_v1_segformer_b1.py']

model = dict(
    tower_head=dict(
        type='TowerHead',
        in_channels=128,
        channels=128,
        loss_bce_weight=1.0,
        loss_dice_weight=0.5,
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'fusion': dict(lr_mult=10.0),
            'center_head': dict(lr_mult=10.0),
            'orient_head': dict(lr_mult=10.0),
            'tower_head': dict(lr_mult=10.0),
        }
    ),
)

work_dir = './work_dirs/powerline_v1_segformer_b1_aux_tower'
