import os
import numpy as np
import matplotlib.pyplot as plt

from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine.config import Config
from mmseg.registry import DATASETS

SAVE_DIR = 'debug_vis'
os.makedirs(SAVE_DIR, exist_ok=True)

cfg = Config.fromfile('configs/powerline_v1/powerline_v1_r18_fpn.py')
dataset = DATASETS.build(cfg.train_dataloader['dataset'])

sample = dataset[0]

# inputs: [3, H, W] -> [H, W, 3]
img = sample['inputs'].numpy().transpose(1, 2, 0)
img = np.clip(img, 0, 255).astype(np.uint8)

data_sample = sample['data_samples']

# center: [H, W]
center = data_sample.gt_sem_seg.data.numpy()[0]

# orient: [2, H, W]
orient = data_sample.gt_orient_map.data.numpy()
ox = orient[0]
oy = orient[1]

print('img.shape =', img.shape)
print('center.shape =', center.shape)
print('orient.shape =', orient.shape)
print('center positive pixels =', int((center > 0).sum()))

# 打印一些中心线像素上的方向向量
ys_all, xs_all = np.where(center > 0)
if len(xs_all) > 0:
    print('\nSample orientation vectors on centerline pixels:')
    step_print = max(1, len(xs_all) // 10)
    for k in range(0, len(xs_all), step_print):
        y, x = ys_all[k], xs_all[k]
        print(f'({x}, {y}) -> ox={ox[y, x]:.4f}, oy={oy[y, x]:.4f}')
else:
    print('Warning: no positive centerline pixels found in this sample.')

# -------------------------
# 从真实中心线像素中抽稀采样
# -------------------------
sample_stride = 1000   # 越大箭头越少，可改 40 / 60 / 100
arrow_scale_vis = 500.0  # 纯可视化倍率，建议 20~40 之间试

if len(xs_all) > 0:
    idx = np.arange(0, len(xs_all), sample_stride)
    xs_draw = xs_all[idx]
    ys_draw = ys_all[idx]

    # GT 是单位向量；显示时放大，否则几乎只有 1 像素长
    u = ox[ys_draw, xs_draw] * arrow_scale_vis
    v = oy[ys_draw, xs_draw] * arrow_scale_vis   
else:
    xs_draw = np.array([], dtype=np.int32)
    ys_draw = np.array([], dtype=np.int32)
    u = np.array([], dtype=np.float32)
    v = np.array([], dtype=np.float32)

print('num arrows to draw =', len(xs_draw))
print('arrow_scale_vis =', arrow_scale_vis)

plt.figure(figsize=(18, 6))

# -------------------------
# 1) 原图
# -------------------------
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('image')
plt.axis('off')

# -------------------------
# 2) 中心线叠加
# -------------------------
plt.subplot(1, 3, 2)
plt.imshow(img)
plt.imshow(center, alpha=0.5, cmap='Reds')
plt.title('centerline overlay')
plt.axis('off')

# -------------------------
# 3) 方向场叠加
# -------------------------
plt.subplot(1, 3, 3)
plt.imshow(img)
plt.imshow(center, alpha=0.20, cmap='gray')

if len(xs_draw) > 0:
    plt.quiver(
        xs_draw,
        ys_draw,
        u,
        v,
        color='yellow',
        angles='xy',
        scale_units='xy',
        scale=1,
        pivot='mid',
        width=0.0025,
        headwidth=4,
        headlength=6,
        headaxislength=5,
    )

plt.title('orientation field')
plt.axis('off')

plt.tight_layout()
save_path = os.path.join(SAVE_DIR, 'dataset_sample_fixed3.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print('\nSaved to', save_path)
