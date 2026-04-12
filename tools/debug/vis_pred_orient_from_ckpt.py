from pathlib import Path
import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# -------------------------
# Make repo root importable
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint
from mmseg.registry import DATASETS, MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize pred center/orient/raw_orient_norm from checkpoint')
    parser.add_argument(
        'config',
        help='Path to config file, e.g. configs/powerline_v1/powerline_v1_r18_fpn_full.py')
    parser.add_argument(
        'checkpoint',
        help='Path to checkpoint, e.g. work_dirs/powerline_v1_r18_fpn/iter_16000.pth')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='work_dirs/powerline_v1_r18_fpn/pred_orient_vis',
        help='Directory to save visualization images')
    parser.add_argument(
        '--dataset-key',
        type=str,
        default='test',
        choices=['test', 'train'],
        help='Which dataloader dataset to visualize')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=20,
        help='Max number of samples to visualize')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Start dataset index')
    parser.add_argument(
        '--pred-thr',
        type=float,
        default=0.3,
        help='Threshold for predicted center mask')
    parser.add_argument(
        '--gt-arrow-stride',
        type=int,
        default=120,
        help='Sample stride for GT orient arrows')
    parser.add_argument(
        '--pred-arrow-stride',
        type=int,
        default=120,
        help='Sample stride for Pred orient arrows')
    parser.add_argument(
        '--arrow-scale',
        type=float,
        default=28.0,
        help='Visualization scale for arrows')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device for inference')
    return parser.parse_args()


def sample_mask_points(mask: np.ndarray, stride: int = 80):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    idx = np.arange(0, len(xs), max(1, stride))
    return xs[idx], ys[idx]


def draw_orient_arrows(
    ax,
    orient_2chw: np.ndarray,
    mask_hw: np.ndarray,
    stride: int,
    arrow_scale: float,
    color: str = 'yellow',
):
    """
    orient_2chw: [2, H, W]
    mask_hw:     [H, W], >0 means valid
    """
    xs, ys = sample_mask_points(mask_hw, stride=stride)
    if len(xs) == 0:
        return

    ox = orient_2chw[0]
    oy = orient_2chw[1]

    u = ox[ys, xs] * arrow_scale
    v = oy[ys, xs] * arrow_scale   # 这里明确使用 oy，不取负号

    ax.quiver(
        xs,
        ys,
        u,
        v,
        color=color,
        angles='xy',
        scale_units='xy',
        scale=1,
        pivot='mid',
        width=0.003,
        headwidth=4,
        headlength=6,
        headaxislength=5,
    )


def build_dataset_from_cfg(cfg, dataset_key='test'):
    if dataset_key == 'test':
        if cfg.get('test_dataloader', None) is not None:
            dataset_cfg = cfg.test_dataloader['dataset']
        else:
            dataset_cfg = cfg.train_dataloader['dataset']
    else:
        dataset_cfg = cfg.train_dataloader['dataset']

    dataset = DATASETS.build(dataset_cfg)
    return dataset


def prepare_batch(model, sample, device):
    """
    Convert one dataset sample into model input through data_preprocessor.
    """
    batch = dict(
        inputs=[sample['inputs']],
        data_samples=[sample['data_samples']],
    )
    processed = model.data_preprocessor(batch, training=False)
    processed['inputs'] = processed['inputs'].to(device)
    return processed


def tensor_to_uint8_image(img_chw: np.ndarray) -> np.ndarray:
    """
    Input from dataset sample['inputs'] before data_preprocessor:
    usually float tensor but pixel range still around 0~255.
    """
    img = np.transpose(img_chw, (1, 2, 0))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def normalize_heatmap(x: np.ndarray):
    x = x.astype(np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin)


def save_one_visualization(
    save_path: Path,
    img: np.ndarray,
    gt_center: np.ndarray,
    pred_center_prob: np.ndarray,
    pred_center_bin: np.ndarray,
    gt_orient: np.ndarray,
    pred_orient: np.ndarray,
    raw_orient_norm: np.ndarray,
    gt_arrow_stride: int,
    pred_arrow_stride: int,
    arrow_scale: float,
    pred_thr: float,
    img_name: str,
):
    raw_orient_norm_vis = normalize_heatmap(raw_orient_norm)
    raw_orient_norm_on_pred = raw_orient_norm_vis * pred_center_bin.astype(np.float32)

    fig, axes = plt.subplots(2, 5, figsize=(30, 12))

    # 1 raw image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('raw image')
    axes[0, 0].axis('off')

    # 2 GT center
    axes[0, 1].imshow(img)
    axes[0, 1].imshow(gt_center, alpha=0.45, cmap='Reds')
    axes[0, 1].set_title('GT center')
    axes[0, 1].axis('off')

    # 3 Pred center prob
    axes[0, 2].imshow(img)
    axes[0, 2].imshow(pred_center_prob, alpha=0.55, cmap='viridis')
    axes[0, 2].set_title(f'Pred center prob (thr={pred_thr:.2f})')
    axes[0, 2].axis('off')

    # 4 GT + Pred center compare
    axes[0, 3].imshow(img)
    axes[0, 3].imshow(gt_center, alpha=0.35, cmap='Reds')
    axes[0, 3].imshow(pred_center_bin, alpha=0.35, cmap='Blues')
    axes[0, 3].set_title('GT center (red) + Pred center (blue)')
    axes[0, 3].axis('off')

    # 5 raw orient norm
    axes[0, 4].imshow(img)
    axes[0, 4].imshow(raw_orient_norm_vis, alpha=0.60, cmap='magma')
    axes[0, 4].set_title('Raw orient norm')
    axes[0, 4].axis('off')

    # 6 GT orient on GT center
    axes[1, 0].imshow(img)
    axes[1, 0].imshow(gt_center, alpha=0.18, cmap='gray')
    draw_orient_arrows(
        axes[1, 0],
        orient_2chw=gt_orient,
        mask_hw=gt_center,
        stride=gt_arrow_stride,
        arrow_scale=arrow_scale,
        color='yellow',
    )
    axes[1, 0].set_title('GT orient on GT center')
    axes[1, 0].axis('off')

    # 7 Pred orient on GT center
    axes[1, 1].imshow(img)
    axes[1, 1].imshow(gt_center, alpha=0.18, cmap='gray')
    draw_orient_arrows(
        axes[1, 1],
        orient_2chw=pred_orient,
        mask_hw=gt_center,
        stride=pred_arrow_stride,
        arrow_scale=arrow_scale,
        color='cyan',
    )
    axes[1, 1].set_title('Pred orient on GT center')
    axes[1, 1].axis('off')

    # 8 Pred orient on Pred center
    axes[1, 2].imshow(img)
    axes[1, 2].imshow(pred_center_bin, alpha=0.18, cmap='gray')
    draw_orient_arrows(
        axes[1, 2],
        orient_2chw=pred_orient,
        mask_hw=pred_center_bin,
        stride=pred_arrow_stride,
        arrow_scale=arrow_scale,
        color='lime',
    )
    axes[1, 2].set_title('Pred orient on Pred center')
    axes[1, 2].axis('off')

    # 9 Pred center binary
    axes[1, 3].imshow(img)
    axes[1, 3].imshow(pred_center_bin, alpha=0.55, cmap='Blues')
    axes[1, 3].set_title('Pred center binary')
    axes[1, 3].axis('off')

    # 10 raw orient norm on Pred center
    axes[1, 4].imshow(img)
    axes[1, 4].imshow(raw_orient_norm_on_pred, alpha=0.65, cmap='magma')
    axes[1, 4].set_title('Raw orient norm on Pred center')
    axes[1, 4].axis('off')

    plt.suptitle(img_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(cfg_path))
    dataset = build_dataset_from_cfg(cfg, dataset_key=args.dataset_key)

    print('Repo root =', ROOT)
    print('Config =', cfg_path)
    print('Checkpoint =', ckpt_path)
    print('Dataset key =', args.dataset_key)
    print('Dataset length =', len(dataset))
    print('Output dir =', out_dir)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(ckpt_path), map_location='cpu')
    model.to(args.device)
    model.eval()

    end_idx = min(len(dataset), args.start_idx + args.max_samples)

    for idx in range(args.start_idx, end_idx):
        sample = dataset[idx]

        img_path = sample['data_samples'].metainfo.get('img_path', f'idx_{idx}')
        img_name = Path(img_path).name

        # raw image from dataset sample
        img = tensor_to_uint8_image(sample['inputs'].numpy())

        # GT
        gt_center = sample['data_samples'].gt_sem_seg.data.numpy()[0].astype(np.uint8)
        gt_orient = sample['data_samples'].gt_orient_map.data.numpy().astype(np.float32)

        processed = prepare_batch(model, sample, args.device)

        with torch.no_grad():
            # 直接拿 raw outputs
            pred_dict = model._forward(processed['inputs'], processed['data_samples'])

            center_logits = pred_dict['center_logits']          # [1, 1, h, w]
            raw_orient_pred = pred_dict['orient_pred']         # [1, 2, h, w]

            input_h, input_w = processed['inputs'].shape[-2:]

            center_logits_up = F.interpolate(
                center_logits,
                size=(input_h, input_w),
                mode='bilinear',
                align_corners=False,
            )

            raw_orient_pred_up = F.interpolate(
                raw_orient_pred,
                size=(input_h, input_w),
                mode='bilinear',
                align_corners=False,
            )

            center_prob = torch.sigmoid(center_logits_up)      # [1, 1, H, W]
            pred_center_prob = center_prob[0, 0].detach().cpu().numpy().astype(np.float32)
            pred_center_bin = (pred_center_prob > args.pred_thr).astype(np.uint8)

            pred_orient_unit = model.orient_head.normalize_to_unit_vector(
                raw_orient_pred_up
            )[0].detach().cpu().numpy().astype(np.float32)    # [2, H, W]

            raw_orient_norm = torch.norm(
                raw_orient_pred_up, dim=1
            )[0].detach().cpu().numpy().astype(np.float32)    # [H, W]

        save_path = out_dir / f'{idx:04d}_{Path(img_name).stem}.png'
        save_one_visualization(
            save_path=save_path,
            img=img,
            gt_center=gt_center,
            pred_center_prob=pred_center_prob,
            pred_center_bin=pred_center_bin,
            gt_orient=gt_orient,
            pred_orient=pred_orient_unit,
            raw_orient_norm=raw_orient_norm,
            gt_arrow_stride=args.gt_arrow_stride,
            pred_arrow_stride=args.pred_arrow_stride,
            arrow_scale=args.arrow_scale,
            pred_thr=args.pred_thr,
            img_name=img_name,
        )

        print(f'[{idx + 1}/{end_idx}] saved -> {save_path}')

    print('Done.')


if __name__ == '__main__':
    main()
