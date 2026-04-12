from mmengine.config import Config
from mmseg.registry import MODELS, DATASETS, TRANSFORMS

cfg = Config.fromfile('configs/powerline_v1/powerline_v1_r18_fpn.py')

print('Config loaded.')
print('Model type:', cfg.model.type)

# Try build dataset config only
dataset_cfg = cfg.train_dataloader['dataset']
print('Dataset type:', dataset_cfg['type'])

print('Registered check passed.')