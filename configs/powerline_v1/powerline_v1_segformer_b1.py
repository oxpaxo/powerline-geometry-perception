default_scope = 'mmseg'

_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['projects.powerline_v1'],
    allow_failed_imports=False,
)

# Dataset root

data_root = './projects/powerline_v1/datasets/TTPLA'

# Patch training size
crop_size = (512, 1024)

# Official MMSeg SegFormer-B1 pretrained checkpoint
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'

model = dict(
    type='PowerLineSegmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=0,
        size=crop_size,
    ),
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    ),
    neck=None,
    fusion=dict(
        type='SharedFusion',
        in_channels_list=[64, 128, 320, 512],
        out_channels=128,
        num_levels=4,
    ),
    center_head=dict(
        type='CenterlineHead',
        in_channels=128,
        channels=128,
    ),
    orient_head=dict(
        type='OrientationHead',
        in_channels=128,
        channels=128,
        loss_smoothl1_weight=1.0,
        loss_cosine_weight=0.2,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        center_threshold=0.3,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPowerLineAnnotations'),
    dict(
        type='PowerLineRandomCrop',
        crop_size=crop_size,
        min_positive_pixels=64,
        max_try=10,
        pad_if_needed=True,
    ),
    dict(
        type='PowerLineRandomFlip',
        prob=0.5,
        direction='horizontal',
    ),
    dict(type='PackPowerLineInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPowerLineAnnotations'),
    dict(type='PackPowerLineInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='PowerLineDataset',
        data_root=data_root,
        ann_file='splits/train.txt',
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='center/train',
            orient_path='orient/train',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = None

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PowerLineDataset',
        data_root=data_root,
        ann_file='splits/val.txt',
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='center/val',
            orient_path='orient/val',
        ),
        pipeline=test_pipeline,
    ),
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=2000,
)

val_cfg = None
test_cfg = dict(type='TestLoop')
val_evaluator = None

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'fusion': dict(lr_mult=10.0),
            'center_head': dict(lr_mult=10.0),
            'orient_head': dict(lr_mult=10.0),
        }
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500,
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    ),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False),
)

env_cfg = dict(
    cudnn_benchmark=True,
)

log_level = 'INFO'
load_from = None
resume = False

work_dir = './work_dirs/powerline_v1_segformer_b1'
