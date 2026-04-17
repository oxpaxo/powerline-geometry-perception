_base_ = ['./powerline_v1_segformer_b1.py']

model = dict(
    distance_head=dict(
        type='DistanceFieldHead',
        in_channels=128,
        channels=64,
        loss_distance_weight=0.5,
        target_normalize_mode='clip_max',
        max_distance_clip=50.0,
        use_mask=True,
        loss_type='smooth_l1',
    ),
    train_cfg=dict(
        patch_global_teacher_cfg=dict(
            num_theta_bins=12,
            num_rho_bins=64,
            angle_smooth_sigma_bins=1.25,
            direction_peak_kernel_sigma_deg=8.0,
            prior_blur_kernel=31,
            prior_dilate_kernel=9,
            prior_downsample_stride=1,
            min_center_pixels=8,
            cache_full_gt=True,
        ),
        debug_teacher_stats=False,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPowerLineAnnotations',
        generate_distance_map=True,
    ),
    dict(
        type='PowerLineRandomCrop',
        crop_size=(512, 1024),
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
    dict(
        type='LoadPowerLineAnnotations',
        generate_distance_map=False,
    ),
    dict(type='PackPowerLineInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'fusion': dict(lr_mult=10.0),
            'center_head': dict(lr_mult=10.0),
            'orient_head': dict(lr_mult=10.0),
            'distance_head': dict(lr_mult=10.0),
        }
    ),
)

work_dir = './work_dirs/powerline_v1_segformer_b1_aux_distance'
