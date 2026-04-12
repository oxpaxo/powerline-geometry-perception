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
model.eval()

sample = dataset[0]
inputs = sample['inputs'].unsqueeze(0)   # [1, 3, H, W]
data_samples = [sample['data_samples']]

print('inputs.shape =', tuple(inputs.shape))
print('gt_sem_seg.shape =', tuple(data_samples[0].gt_sem_seg.data.shape))
print('gt_orient_map.shape =', tuple(data_samples[0].gt_orient_map.data.shape))

with torch.no_grad():
    results = model.predict(inputs, data_samples)

pred = results[0]

print('\nPrediction fields:')
print('pred_sem_seg:', tuple(pred.pred_sem_seg.data.shape))
print('seg_logits:', tuple(pred.seg_logits.data.shape))
print('pred_orient_map:', tuple(pred.pred_orient_map.data.shape))

print('\nPredict check passed.')
