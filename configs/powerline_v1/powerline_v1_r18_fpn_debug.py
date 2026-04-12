# configs/powerline_v1/powerline_v1_r18_fpn.py

default_scope = 'mmseg'

_base_ = [
    "../_base_/default_runtime.py",
]

# --------------------
# 1) Basic settings
# --------------------
default_scope = "mmseg"

custom_imports = dict(
    imports=["projects.powerline_v1"],
    allow_failed_imports=False,
)

# Replace this with your real dataset root
data_root = "./projects/powerline_v1/datasets/TTPLA"

# Original image size is 3840x2160, but V1 uses patch training.
# Each training sample is cropped from the large image.
crop_size = (512, 1024)  # (crop_h, crop_w)

# -----------
# 2) Model
# -----------
model = dict(
    type="PowerLineSegmentor",
    # data_preprocessor=dict(
    #     type="SegDataPreProcessor",
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     bgr_to_rgb=True,
    #     pad_val=0,
    #     seg_pad_val=0,
    # ),
    data_preprocessor=dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size,
), # 以后做不固定尺寸、整图或滑窗验证,比如size_divisor=32
    backbone=dict(
        type="ResNet",
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
    ),
    neck=dict(
        type="FPN",
        in_channels=[64, 128, 256, 512],
        out_channels=128,
        num_outs=4,
    ),
    fusion=dict(
        type="SharedFusion",
        in_channels=128,
        num_levels=4,
        out_channels=128,
    ),
    center_head=dict(
        type="CenterlineHead",
        in_channels=128,
        channels=128,
    ),
    orient_head=dict(
        type="OrientationHead",
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

# ---------------
# 3) Pipelines
# ---------------
# train_pipeline = [
#     dict(type="LoadImageFromFile"),
#     dict(type="LoadPowerLineAnnotations"),
#     dict(
#         type="PowerLineRandomCrop",
#         crop_size=crop_size,
#         min_positive_pixels=64,
#         max_try=10,
#         pad_if_needed=True,
#     ),
#     dict(
#         type="PowerLineRandomFlip",
#         prob=0.5,
#         direction="horizontal",
#     ),
#     dict(type="PhotoMetricDistortion"),
#     dict(type="PackPowerLineInputs"),
# ]
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadPowerLineAnnotations"),
    dict(
        type="PowerLineRandomCrop",
        crop_size=crop_size,
        min_positive_pixels=64,
        max_try=10,
        pad_if_needed=True,
    ),
    dict(
        type="PowerLineRandomFlip",
        prob=0.5,
        direction="horizontal",
    ),
    dict(type="PackPowerLineInputs"),
]


# For now, this config is focused on training only.
# Full-image validation for 3840x2160 can be added later with sliding-window / tiling.
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadPowerLineAnnotations"),
    dict(type="PackPowerLineInputs"),
]

# -----------------
# 4) DataLoaders
# -----------------
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type="InfiniteSampler", shuffle=True),
#     dataset=dict(
#         type="PowerLineDataset",
#         data_root=data_root,
#         ann_file="splits/train.txt",
#         data_prefix=dict(
#             img_path="images/train",
#             seg_map_path="center/train",
#             orient_path="orient/train",
#         ),
#         pipeline=train_pipeline,
#     ),
# )
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="PowerLineDataset",
        data_root=data_root,
        ann_file="splits/debug_train.txt",
        data_prefix=dict(
            img_path="images/train",
            seg_map_path="center/train",
            orient_path="orient/train",
        ),
        pipeline=train_pipeline,
    ),
)

# Keep validation disabled in the first debug stage.
val_dataloader = None
# test_dataloader = None
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PowerLineDataset',
        data_root=data_root,
        ann_file='splits/debug_train.txt',
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='center/train',
            orient_path='orient/train',
        ),
        pipeline=test_pipeline,
    ),
)

# ---------------------------------
# 5) Training / Evaluation loops
# ---------------------------------
# train_cfg = dict(
#     type="IterBasedTrainLoop",
#     max_iters=20000,
#     val_interval=2000,
# )
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=500,
    val_interval=1000,
)

val_cfg = None
# test_cfg = None
# val_evaluator = None
# test_evaluator = None
test_cfg = dict(type='TestLoop')
val_evaluator = None

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
)

# ---------------
# 6) Optimizer
# ---------------
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=1e-4,
    ),
)

# ------------------
# 7) LR scheduler
# ------------------
param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=500,#20000
        by_epoch=False,
    )
]

# ---------------------
# 8) Hooks / runtime
# ---------------------
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=2000, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook", draw=False),
)

env_cfg = dict(
    cudnn_benchmark=True,
)

log_level = "INFO"
load_from = None
resume = False

work_dir = "./work_dirs/powerline_v1_r18_fpn"
