_base_ = ['./powerline_v1_segformer_b1_aux_distance.py']

# Sliding-window inference for large TTPLA images.
model = dict(
    test_cfg=dict(
        mode='slide',
        crop_size=(512, 1024),
        stride=(384, 768),
        center_threshold=0.3,
        use_geometric_verifier=True,
        verifier_cfg=dict(
            morph_open_kernel=3,
            morph_close_kernel=7,
            min_area=600,
            min_length=300.0,
            min_aspect_ratio=3.5,
            min_center_mean=0.40,
            min_orient_consistency=0.9,
            use_distance=True,
            max_distance_mean=0.20,
            max_distance_q80=0.25,
            dominant_bin_deg=12.0, ####
            dominant_keep_topk=3, ####
            dominant_angle_tol_deg=9.0,
            strong_length_px=500.0,
            short_length_px=350.0,
            max_isolated_dist_px=70.0, ####
            debug=True,
        ),
    )
)

# Keep test batch size = 1.
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
)