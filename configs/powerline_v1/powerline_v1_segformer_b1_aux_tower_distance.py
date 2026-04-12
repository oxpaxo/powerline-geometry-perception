"""
Combined distance + tower auxiliary heads config.

Distance head: active (gt_distance_map generated online from center mask).
Tower head: PLACEHOLDER — no gt_tower_seg available yet; tower loss is
            silently skipped at runtime. See aux_tower config for details.
"""
_base_ = ['./powerline_v1_segformer_b1_aux_distance.py']

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
            'distance_head': dict(lr_mult=10.0),
            'tower_head': dict(lr_mult=10.0),
        }
    ),
)

work_dir = './work_dirs/powerline_v1_segformer_b1_aux_tower_distance'
