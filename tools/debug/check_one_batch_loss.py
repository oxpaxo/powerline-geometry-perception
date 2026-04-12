from pathlib import Path
import sys
import os

import torch

# -------------------------
# Make repo root importable
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional: force current working directory to repo root
os.chdir(ROOT)

from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine.config import Config
from mmseg.registry import DATASETS, MODELS

CFG_PATH = ROOT / 'configs' / 'powerline_v1' / 'powerline_v1_r18_fpn.py'

cfg = Config.fromfile(str(CFG_PATH))

print('Repo root =', ROOT)
print('Config =', CFG_PATH)

dataset = DATASETS.build(cfg.train_dataloader['dataset'])
print('Dataset length =', len(dataset))

model = MODELS.build(cfg.model)
model.train()

# 先拿两个样本测试
sample0 = dataset[0]
sample1 = dataset[1]

inputs = torch.stack([sample0['inputs'], sample1['inputs']], dim=0)  # [N, 3, H, W]
data_samples = [sample0['data_samples'], sample1['data_samples']]

print('inputs.shape =', tuple(inputs.shape))
print('sample0 gt_sem_seg =', tuple(sample0['data_samples'].gt_sem_seg.data.shape))
print('sample0 gt_orient_map =', tuple(sample0['data_samples'].gt_orient_map.data.shape))
print('sample1 gt_sem_seg =', tuple(sample1['data_samples'].gt_sem_seg.data.shape))
print('sample1 gt_orient_map =', tuple(sample1['data_samples'].gt_orient_map.data.shape))

losses = model.loss(inputs, data_samples)

print('\nLoss dict:')
for k, v in losses.items():
    if torch.is_tensor(v):
        print(f'{k}: {float(v.detach().cpu())}')
    else:
        print(f'{k}: {v}')

print('\nOne-batch loss check passed.')
