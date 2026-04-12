"""
Inspect auxiliary targets and predictions.

Usage:
    # Random init model, visualize GT targets only:
    python tools/debug/inspect_aux_targets_and_preds.py \
        --config configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py \
        --num-samples 4 \
        --out-dir work_dirs/debug_vis

    # With checkpoint, visualize both GT and predictions:
    python tools/debug/inspect_aux_targets_and_preds.py \
        --config configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py \
        --checkpoint work_dirs/.../iter_20000.pth \
        --num-samples 4 \
        --out-dir work_dirs/debug_vis
"""

import argparse
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect auxiliary targets and model predictions')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint file. If None, uses random init.')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--out-dir', default='work_dirs/debug_vis',
                        help='Output directory for visualization images')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for inference')
    return parser.parse_args()


def colorize_distance(dist_map, max_val=None):
    """Convert distance map to colorized heatmap (H,W) -> (H,W,3) uint8."""
    if max_val is None:
        max_val = dist_map.max() if dist_map.max() > 0 else 1.0
    normed = np.clip(dist_map / max_val, 0, 1)
    cmap = plt.cm.viridis(normed)[:, :, :3]
    return (cmap * 255).astype(np.uint8)


def colorize_orient_hsv(orient_map, mask=None):
    """Convert orientation [2,H,W] to HSV visualization (H,W,3) uint8."""
    ox = orient_map[0]
    oy = orient_map[1]
    angle = np.arctan2(oy, ox)
    angle_deg = np.degrees(angle) % 360
    mag = np.sqrt(ox ** 2 + oy ** 2)
    mag_norm = np.clip(mag / (mag.max() + 1e-6), 0, 1)

    hsv = np.zeros((*ox.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (angle_deg / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)

    if mask is not None:
        hsv[mask <= 0, 1] = 0
        hsv[mask <= 0, 2] = 0

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_panel(images, titles, save_path, suptitle=''):
    """Save a row of images as a single figure."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def get_field_numpy(sample, field_name):
    """Safely get a PixelData field as numpy array, or return None."""
    if not hasattr(sample, field_name):
        return None
    field = getattr(sample, field_name)
    if field is None or not hasattr(field, 'data'):
        return None
    return field.data.cpu().numpy()


def to_display_image(inputs_chw: torch.Tensor, bgr_to_rgb: bool = True) -> np.ndarray:
    """
    Convert packed dataset input [3,H,W] to displayable RGB uint8 image.

    Important:
    - dataset['inputs'] here is the raw packed tensor from PackPowerLineInputs
    - it has NOT gone through model.data_preprocessor yet
    """
    img = inputs_chw.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)

    if bgr_to_rgb and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def build_inference_batch(model, inputs, sample):
    """
    Run the same preprocessing path as real inference.

    Returns:
        processed_inputs: batched tensor [N,C,H,W]
        batch_img_metas: list[dict] or None
    """
    data_batch = dict(
        inputs=[inputs],
        data_samples=[sample],
    )

    processed = model.data_preprocessor(data_batch, training=False)
    processed_inputs = processed['inputs']

    processed_samples = processed.get('data_samples', None)
    batch_img_metas = None
    if processed_samples is not None:
        batch_img_metas = [s.metainfo for s in processed_samples]

    return processed_inputs, batch_img_metas


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)

    with DefaultScope.overwrite_default_scope(cfg.get('default_scope', 'mmseg')):
        from mmseg.registry import DATASETS, MODELS

        train_ds_cfg = cfg.train_dataloader.dataset
        dataset = DATASETS.build(train_ds_cfg)
        print(f'Dataset loaded: {len(dataset)} samples')

        model = MODELS.build(cfg.model)
        if args.checkpoint:
            from mmengine.runner import load_checkpoint
            load_checkpoint(model, args.checkpoint, map_location='cpu')
            print(f'Loaded checkpoint: {args.checkpoint}')
        else:
            print('Using random initialized model')

        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        bgr_to_rgb = bool(cfg.model.data_preprocessor.get('bgr_to_rgb', False))

        num = min(args.num_samples, len(dataset))
        for idx in range(num):
            data = dataset[idx]
            inputs = data['inputs']           # [3, H, W], raw packed tensor
            sample = data['data_samples']     # SegDataSample

            img_show = to_display_image(inputs, bgr_to_rgb=bgr_to_rgb)

            images = [img_show]
            titles = ['Input']

            gt_center = get_field_numpy(sample, 'gt_sem_seg')
            if gt_center is not None:
                gt_center_vis = (gt_center[0] * 255).astype(np.uint8)
                images.append(gt_center_vis)
                titles.append('GT Center')

            gt_orient = get_field_numpy(sample, 'gt_orient_map')
            if gt_orient is not None:
                gt_center_mask = gt_center[0] if gt_center is not None else None
                orient_vis = colorize_orient_hsv(gt_orient, gt_center_mask)
                images.append(orient_vis)
                titles.append('GT Orient (HSV)')

            gt_dist = get_field_numpy(sample, 'gt_distance_map')
            if gt_dist is not None:
                dist_vis = colorize_distance(gt_dist[0], max_val=50.0)
                images.append(dist_vis)
                titles.append('GT Distance')

            gt_tower = get_field_numpy(sample, 'gt_tower_seg')
            if gt_tower is not None:
                tower_vis = (gt_tower[0] * 255).astype(np.uint8)
                images.append(tower_vis)
                titles.append('GT Tower')

            with torch.no_grad():
                processed_inputs, batch_img_metas = build_inference_batch(
                    model, inputs, sample
                )
                processed_inputs = processed_inputs.to(device)
                pred_dict = model.encode_decode(
                    processed_inputs,
                    batch_img_metas=batch_img_metas,
                )

            h, w = img_show.shape[:2]

            center_prob_up = None
            if 'center_logits' in pred_dict:
                center_prob = torch.sigmoid(pred_dict['center_logits'])
                center_prob_np = center_prob[0, 0].cpu().numpy()
                center_prob_up = cv2.resize(
                    center_prob_np,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                images.append((center_prob_up * 255).astype(np.uint8))
                titles.append('Pred Center Prob')

            if 'orient_pred' in pred_dict:
                orient_pred = pred_dict['orient_pred'][0].cpu().numpy()
                norm = np.sqrt(orient_pred[0] ** 2 + orient_pred[1] ** 2) + 1e-6
                orient_pred_unit = orient_pred / norm

                ox_up = cv2.resize(
                    orient_pred_unit[0],
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                oy_up = cv2.resize(
                    orient_pred_unit[1],
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                orient_pred_up = np.stack([ox_up, oy_up], axis=0)

                pred_center_mask = (center_prob_up > 0.3) if center_prob_up is not None else None
                orient_pred_vis = colorize_orient_hsv(
                    orient_pred_up,
                    pred_center_mask,
                )
                images.append(orient_pred_vis)
                titles.append('Pred Orient (HSV)')

            if 'distance_pred' in pred_dict:
                dist_pred = pred_dict['distance_pred'][0, 0].cpu().numpy()
                dist_pred_up = cv2.resize(
                    dist_pred,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                # distance_pred is usually normalized to [0,1]-like range
                dist_pred_vis = colorize_distance(dist_pred_up, max_val=1.0)
                images.append(dist_pred_vis)
                titles.append('Pred Distance')

            if 'tower_pred' in pred_dict:
                tower_pred = torch.sigmoid(pred_dict['tower_pred'])
                tower_pred_np = tower_pred[0, 0].cpu().numpy()
                tower_pred_up = cv2.resize(
                    tower_pred_np,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                images.append((tower_pred_up * 255).astype(np.uint8))
                titles.append('Pred Tower Prob')

            stem = f'sample_{idx:04d}'
            meta = sample.metainfo
            if 'img_path' in meta:
                stem = osp.splitext(osp.basename(meta['img_path']))[0]

            save_path = osp.join(args.out_dir, f'{stem}_panel.png')
            save_panel(images, titles, save_path, suptitle=stem)

    print(f'\nDone. {num} panels saved to {args.out_dir}')


if __name__ == '__main__':
    main()