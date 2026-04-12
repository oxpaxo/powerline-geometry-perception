_base_ = ['./powerline_v1_segformer_b1_aux_tower_distance.py']

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        ann_file='splits/debug_train.txt',
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='center/train',
            orient_path='orient/train',
        ),
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        ann_file='splits/debug_train.txt',
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='center/train',
            orient_path='orient/train',
        ),
    ),
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=500,
    val_interval=1000,
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=50,
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=50,
        end=500,
        by_epoch=False,
    ),
]

work_dir = './work_dirs/powerline_v1_segformer_b1_aux_tower_distance_debug'
